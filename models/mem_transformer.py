import torch
from torch.nn import functional as F
from torch import nn
from einops import rearrange, repeat
from torch import Tensor
from typing import Optional
import math

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
                       if project_out else nn.Identity())

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
    

class RelativeMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, d_model)

    def _rel_shift(self, x):
        zero_pad = torch.zeros((*x.size[:-2], 1, *x.size()[-2:]), device=x.device, dtype=x.type)
        x_padded = torch.cat([zero_pad, x], dim=2)
        x_padded = x_padded.view(*x.size()[:-2], x.size(-2) + 1, x.size(-1))

        x = x_padded[:, 1:].view_as(x)
        return x
    
    