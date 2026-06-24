import os
import pdb
import time
import torch
# import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        # self.ctc_decoder = ctcdecode.CTCBeamDecoder(vocab, beam_width=10, blank_id=blank_id,
        #                                             num_processes=10)

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)])
        return ret_list


    # def BeamSearch(self, nn_output, vid_lgt, probs=False):
    #     '''
    #     CTCBeamDecoder Shape:
    #             - Input:  nn_output (B, T, N), which should be passed through a softmax layer
    #             - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
    #                       beam_scores (B, N_beams), p=1/np.exp(beam_score)
    #                       timesteps (B, N_beams)
    #                       out_lens (B, N_beams)
    #     '''
    #     if not probs:
    #         nn_output = nn_output.softmax(-1).cpu()
    #     vid_lgt = vid_lgt.cpu()
    #     beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
    #     ret_list = []
    #     for batch_idx in range(len(nn_output)):
    #         first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
    #         if len(first_result) != 0:
    #             first_result = torch.stack([x[0] for x in groupby(first_result)])
    #         ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
    #                          enumerate(first_result)])
    #     return ret_list
    #

    from itertools import groupby

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []

        for batch_idx in range(batchsize):
            # ✅【核心修复】确保 length 是 普通整数
            length = int(vid_lgt[batch_idx].item())

            # ✅【核心修复】安全切片
            sequence = index_list[batch_idx][:length]

            # 转 list 避免 tensor 不兼容问题
            sequence = sequence.cpu().tolist()

            # CTC 去重
            group_result = [x[0] for x in groupby(sequence)]

            # 去掉 blank
            filtered = [x for x in group_result if x != self.blank_id]

            # 再次去重（兼容原逻辑）
            if len(filtered) > 0:
                max_result = [x[0] for x in groupby(filtered)]
            else:
                max_result = filtered

            # 转词汇
            ret_list.append([
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(max_result)
            ])

        return ret_list
