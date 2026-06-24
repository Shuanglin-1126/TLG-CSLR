import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
from einops import rearrange
from modules.get_patches import get_key_patches

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, num_k=32, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.get_key_patches = get_key_patches(num_k=num_k)
        if '16' in c2d_type:
            self.patch_size = 16
        elif '14' in c2d_type:
            self.patch_size = 14

        # For openai clip
        from modules.openai import clip
        # from modules.openai import huggingface
        self.conv2d = clip.load(c2d_type)

        self.conv1d = TemporalConv(input_size=512 if '16' in c2d_type else 768, #768 for ViT-L/14, 512 for ViT-B/16, 512 for ViT-B/32, 1024 for RN50
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'max')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        #self.fc = NormLinear(1024, self.num_classes)
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, label=None, label_lgt=None):
        with torch.cuda.amp.autocast():
            batch, temp, channel, height, width = x.shape
            num_patch = int(height / self.patch_size)
            x_clone = x.clone()
            x_clone = rearrange(x_clone, 'b t c (n1 s1) (n2 s2) -> b t (n1 n2) (c s1 s2)',
                                n1=num_patch, s1=self.patch_size, n2=num_patch, s2=self.patch_size)
            idx_patches = self.get_key_patches(x_clone).detach()

            framewise, attn_list = self.conv2d.encode_image(x.flatten(0, 1), idx_patches)
            temp -= 1
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2).to(torch.float32)

            len_x = torch.full((batch,), temp, device=x.device)

            #framewise_logits = self.fc(framewise.transpose(1, 2)).permute(1,0,2)
            x, lgt, conv_logtic = self.conv1d(framewise, len_x)
            tm_outputs, _ = self.temporal_model(x, lgt)
            outputs = self.classifier(tm_outputs)
            pred = None if self.training \
                else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
            conv_pred = None if self.training \
                else self.decoder.decode(conv_logtic, lgt, batch_first=False, probs=False)

        return {
            #"framewise_logits": framewise_logits,
            #"visual_features": x,
            "idx":idx_patches,
            "attn": attn_list,
            "feat_len": lgt,
            "conv_logits": conv_logtic,
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].to(torch.float32).log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].to(torch.float32).log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Direct':
                loss += weight * self.loss['CTCLoss'](ret_dict["framewise_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"].to(torch.float32),
                                                           ret_dict["sequence_logits"].to(torch.float32).detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss


if __name__ == '__main__':
    import time
    # from fvcore.nn import FlopCountAnalysis
    # from fvcore.nn import flop_count_table
    import numpy as np

    num_frames = 8

    model = SLRModel(1296, 'ViT-B/16', 2).cuda().train()
    checkpoint = torch.load(r'.\output\phoenix\vitb16\_best_model.pt', weights_only=False)['model_state_dict']
    model.load_state_dict(checkpoint)
    torch.save(model.state_dict(), r'.\output\phoenix\vitb16\best_model_2.pt')
    # curr_state_dict = model.state_dict()
    x = torch.randn(2, 64, 3, 224, 224).cuda()
    y = model(x)
    # print(model)
    print(len(y))