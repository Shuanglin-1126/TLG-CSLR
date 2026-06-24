import torch
import torch.nn as nn


class get_key_patches(nn.Module):
    def __init__(self, dim=768, num_k=32, thresold=100):
        super().__init__()
        self.num_k = num_k
        self.thresold = thresold
        self.weight = nn.Parameter(torch.empty((dim, dim)), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(dim), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        # x = self.proj(x)
        B, T, L, C = x.shape
        dis_t = torch.abs(x[:, 1:, ...] - x[:, :-1, ...])
        dis_t = torch.sum(dis_t, dim=-1)

        dis_s = torch.cdist(x, x, p=1)
        bool_tensor = dis_s < self.thresold
        count = bool_tensor.sum(dim=-1)

        dis = dis_t * torch.exp(-1 * count[:, 1:, :])
        _, topk_idx = torch.topk(dis, k=self.num_k, dim=-1)

        return topk_idx