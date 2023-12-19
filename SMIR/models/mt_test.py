import torch
from torch.nn import functional as F
from torch import nn
from einops import rearrange, repeat
from torch import Tensor
from typing import Optional
import math


class PositionalFusionLayer(nn.Module):
    def __init__(self, feature_dim, pos_dim, fusion_dim):
        super(PositionalFusionLayer, self).__init__()
        
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim + pos_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, feature_dim)
        )

        self.positional_embeddings = nn.Parameter(torch.randn(1, 64, pos_dim))

    def forward(self, x):
        pos_emb = self.positional_embeddings.repeat(x.shape[0], 1, 1)
        x = torch.cat([x, pos_emb], dim=-1)
        x = self.fusion_net(x)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[4096, 1024, 512]):
        super(FeatureExtractor, self).__init__()
        layers = []

        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.5))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MT1(nn.Module): # Music Transformer
    def __init__(self,
                 *,
                 input_dim,
                 output_dim,
                 dim,
                 depth,
                 heads,
                 dropout,
                 mode=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim = dim
        self.mode = mode
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_dim, dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True, dim_feedforward=256, dropout=dropout)
        self.encoder= nn.TransformerEncoder(self.encoder_layer, num_layers=depth)
        self.linear2 = nn.Linear(dim , output_dim)
        self.l2_norm = F.normalize
    

    def mask_pooling(self, x, mask):
        masked_x = torch.mul(x, (~mask).unsqueeze(-1))
        sum_x = torch.sum(masked_x, dim=1)
        avg_x = sum_x / (torch.sum(~mask, dim=1).unsqueeze(-1))
        return avg_x
    
    def forward(self, 
                x: Tensor,
                padding_mask: Optional[Tensor] = None,
                ) -> Tensor:
        x_out = self.relu(self.linear1(x))    
 
        if padding_mask is not None:
            x_out = self.encoder(x_out, src_key_padding_mask=padding_mask)
        else:
            x_out = self.encoder(x_out)
      
        x_out = self.mask_pooling(x_out, padding_mask)
        x_out = self.linear2(x_out)
        if self.mode == 'residual':
            x_out = x_out + self.mask_pooling(x, padding_mask)
        x_out = self.l2_norm(x_out)

        return x_out


if __name__ == "__main__":
    model = MT1(
        input_dim=128,
        output_dim=128,
        dim=256,
        depth=8,
        heads=32,
        dropout=0.1,
    )

    src_padding_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]], dtype=torch.bool)
    print(model(torch.rand(2, 5, 128), src_padding_mask).shape)
