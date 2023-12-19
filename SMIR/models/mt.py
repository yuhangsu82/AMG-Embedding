import torch
from torch.nn import functional as F
from torch import nn
from einops import rearrange, repeat
from torch import Tensor
from typing import Optional
import math


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, scale=0.1, max_len=64):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        position_embed = self.pe[:, :x.size(1)].clone().detach()
        position_embed = position_embed.repeat(x.size(0), 1, 1)
        x = torch.cat([x, position_embed * self.scale], dim=2)
        # x = x + self.scale * position_embed

        return x


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


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MT(nn.Module): # Music Transformer
    def __init__(self,
                 *,
                 input_dim,
                 output_dim,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim = dim

        self.linear1 = nn.Linear(input_dim, dim)
        self.linear2 = nn.Linear(dim, output_dim)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, mask=None):
        x = self.linear1(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.linear2(x)

        return x



if __name__ == "__main__":
    model = MT(
        input_dim=128,
        output_dim=128,
        dim=256,
        depth=10,
        heads=8,
        dim_head=16,
        mlp_dim=256,
        dropout=0.1,
    )

    print(model(torch.rand(2, 4, 128)).shape)
