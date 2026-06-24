from collections import OrderedDict
from typing import Tuple, Union
import math
import numpy as np
import torch
from typing import Optional
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, lora_alpha=1):
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        # A: 从输入到低秩空间
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        # B: 从低秩空间到输出
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 初始化 A 使用 Kaiming Uniform，B 初始化为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # x @ A.T @ B.T * scaling
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, rank=8, lora_alpha=1, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.lora = LoRALayer(in_features, out_features, rank, lora_alpha)

    def forward(self, x):
        # 原始输出 + LoRA 增量
        return super().forward(x) + self.lora(x)


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, d_model: int, n_head: int, attention_dropout=0.1, rank=8, lora_alpha=1):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = n_head
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.prefix_length = 8
        self.prefix_embedding_k = nn.Parameter(torch.randn(1, self.prefix_length, d_model))
        self.prefix_embedding_v = nn.Parameter(torch.randn(1, self.prefix_length, d_model))

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ):
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        keys = torch.cat([self.prefix_embedding_k.repeat(batch_size, 1, 1), keys], dim=1)
        values = torch.cat([self.prefix_embedding_v.repeat(batch_size, 1, 1), values], dim=1)

        queries = queries.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length+self.prefix_length, -1, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length+self.prefix_length, -1, self.head_dim).transpose(1, 2)

        attn = torch.softmax((queries @ keys.transpose(2, 3)) / self.scale, dim=-1)
        out = (attn @ values).transpose(1, 2)

        attn_output = out.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, checkpointing=True):
        super().__init__()

        self.self_attn = CLIPAttention(d_model=d_model, n_head=n_head)
        self.layer_norm1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("fc2", nn.Linear(d_model * 4, d_model))
        ]))
        self.layer_norm2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.checkpointing = checkpointing


    def forward(self, x: torch.Tensor):
        if self.checkpointing:
            attn_ = checkpoint(self.self_attn, checkpoint(self.layer_norm1, x, use_reentrant=False), use_reentrant=False)
            x = x + attn_
        else:
            attn_ = self.self_attn(self.layer_norm1(x))
            x = x + attn_


        if self.checkpointing:
            x = x +\
                checkpoint(self.mlp, checkpoint(self.layer_norm2, x, use_reentrant=False), use_reentrant=False)
        else:
            x = x + self.mlp(self.layer_norm2(x))

        return x



class AggregationBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 1)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 1, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x[:1], x[1:], x[1:], need_weights=False, attn_mask=self.attn_mask)[
            0]  # cls token attends to others

    def forward(self, x: torch.Tensor, cls):
        cls = cls + self.attention(self.ln_1(torch.concat((cls, x), 0)))
        cls = cls + self.mlp(self.ln_2(cls))
        return cls


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size=5, window_stride=1, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x, T):
        # Input shape: (N,C,T,H,W), out: (N,C,T,V*window_size)
        NT, L, D = x.shape
        x = x.view(-1, T, L, D).permute(0, 3, 1, 2)
        x = self.unfold(x)  # (N, C*Window_Size, T*P)
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(-1, D, self.window_size, T, L).permute(0, 3, 1, 2, 4,).reshape(NT, D, -1)  # (NT)C(SP)
        return x


class Correlation_Module(nn.Module):
    def __init__(self, k = 5, nighs = 7):
        super().__init__()
        self.k = k
        self.nighs = nighs
        self.init_decay = -0.1

    def forward(self, x, upfold):
        N, L, D = x.shape
        affinities = x @ upfold.transpose(1, 2) / math.sqrt(D)

        _, indices = torch.topk(affinities, self.k*self.nighs, dim=2)  # (L, k, N, D)
        mask = torch.zeros_like(affinities, dtype=torch.float32)
        mask.scatter_(2, indices, 1.)

        affinities = torch.sigmoid(affinities) * mask / (self.k * self.nighs / 2)  # 非 top-k 的地方自动乘成 0
        features = affinities @ upfold

        return features


class TemporalAggregationBlock(nn.Module):
    def __init__(self, d_model: int, nighs: int = 7, k: int = 9, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = Correlation_Module(k)
        self.ln_1 = LayerNorm(d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model*2)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.upfold = UnfoldTemporalWindows(nighs)
        self.weights = nn.Parameter(torch.tensor([0., 0.]), requires_grad=True)
        self.apply(self.init_weights_xavier)

    def init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            # 使用 nn.init.xavier_uniform_ 初始化权重
            nn.init.xavier_uniform_(m.weight)
            # 偏置通常初始化为 0
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def attention(self, x: torch.Tensor, T):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x = self.linear(x)
        x_upfold = self.upfold(x[..., self.d_model:], T).transpose(1, 2)  # NDL -> LND
        return self.attn(x[..., :self.d_model], x_upfold)

    def forward(self, x: torch.Tensor, T):
        x = x + self.attention(self.ln_1(x), T) * self.weights[0]
        x = x + self.mlp(self.ln_2(x)) * self.weights[1]
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, checkpointing=True):
        super().__init__()
        self.width = width
        self.checkpointing = checkpointing
        self.layers = layers
        if self.checkpointing:
            self.layers = nn.Sequential(
                *[ResidualAttentionBlock(width, heads, attn_mask, checkpointing=i >= 0) for i in range(layers)])
        else:
            self.layers = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

        self.tc_stride = 6
        self.taggblocks = nn.Sequential(*[TemporalAggregationBlock(width, attn_mask=attn_mask) for _ in range(int(layers / self.tc_stride))])

    def forward(self, x: torch.Tensor, T=None):
        B, L, D = x.shape
        for i in range(len(self.layers)):
            x = checkpoint(self.layers[i], x, use_reentrant=False) if self.checkpointing else self.layers[i](x)
            if (i + 1) % self.tc_stride == 0:
                if self.checkpointing:
                    x = x + checkpoint(self.taggblocks[int((i - self.tc_stride + 1) / self.tc_stride)], x, T, use_reentrant=False)
                else:
                    x = x + self.taggblocks[int((i - self.tc_stride + 1) / self.tc_stride)](x, T)
        return x


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int):
        super().__init__()
        self.embed_dim = width
        self.image_size = input_resolution
        self.patch_size = patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        position_embedding = self.position_embedding.weight.unsqueeze(0)
        num_positions = position_embedding.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        class_pos_embed = position_embedding[:, :1]
        patch_pos_embed = position_embedding[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (height != self.image_size or width != self.image_size):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size}*{self.image_size})."
            )
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.embeddings = CLIPVisionEmbeddings(
            input_resolution=input_resolution,
            patch_size=patch_size,
            width=width
        )
        self.pre_layrnorm = LayerNorm(width)

        self.encoder = Transformer(width, layers, heads)

        self.post_layernorm = LayerNorm(width)
        self.visual_projection = nn.Linear(width, output_dim, bias=False)


    def forward(self, x: torch.Tensor, idx_patches):
        # with torch.no_grad():
        x = self.embeddings(x)
        B, T, N = idx_patches.shape
        _, L, C = x.shape
        x = x.view(B, T+1, L, C)

        zeros = torch.zeros(B, T, 1, device=idx_patches.device)
        idx_patches = torch.concat([zeros, idx_patches+1], dim=-1).unsqueeze(-1).expand(-1, -1, -1, C)
        idx_patches = idx_patches
        x = torch.gather(x[:, 1:, ...], 2, idx_patches.long())
        x = x.view(B*T, -1, C)

        x = self.pre_layrnorm(x)
        x = self.encoder(x, T)

        x = self.post_layernorm(x[:, 0, :])

        return self.visual_projection(x), None


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            pass
        else:
            vision_heads = vision_width // 64
            self.vision_model = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.initialize_parameters()

    def initialize_parameters(self):
        pass

    @property
    def dtype(self):
        return self.vision_model.embeddings.patch_embedding.weight.dtype

    def encode_image(self, image, idx_patches):
        return self.vision_model(image, idx_patches)

    def forward(self, image):
        image_features = self.encode_image(image.to(self.dtype))

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def load_matched_state_dict(model, checkpoint, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict:
            if curr_state_dict[key].shape == state_dict[key].shape:
                curr_state_dict[key] = state_dict[key]
                num_matched += 1
        #     else:
        #         print(key)
        # else:
        #     print(key)
    model.load_state_dict(curr_state_dict)
    # if print_stats:
    #     print(f'Loaded state_dict: {num_matched}/{num_total} matched')


def build_model(state_dict: dict):
    vision_width = state_dict["vision_model.pre_layrnorm.weight"].shape[0]
    vision_layers = len(
        [k for k in state_dict.keys() if k.startswith("vision_model.") and k.endswith("self_attn.k_proj.weight")])
    vision_patch_size = state_dict["vision_model.embeddings.patch_embedding.weight"].shape[-1]
    grid_size = round((state_dict["vision_model.embeddings.position_embedding.weight"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["visual_projection.weight"].shape[0]

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size
    )

    # for key in ["input_resolution", "context_length", "vocab_size"]:
    #     if key in state_dict:
    #         del state_dict[key]
    #
    # for key in list(state_dict.keys()):
    #     if 'text' in key and not 'vision' in key:
    #         del state_dict[key]

    # convert_weights(model)
    #model.load_state_dict(state_dict, strict=False)
    load_matched_state_dict(model, state_dict)
    return model