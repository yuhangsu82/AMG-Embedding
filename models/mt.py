import torch
from torch.nn import functional as F
from torch import nn
from torch import Tensor
import time


class MT(nn.Module):
    def __init__(
        self,
        input_dim=128,
        output_dim=128,
        dim=256,
        depth=10,
        heads=8,
        dropout=0.1,
    ):
  
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim = dim
        self.linear1 = nn.Linear(input_dim, dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True,
            dim_feedforward=256,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)
        self.linear2 = nn.Linear(dim, output_dim)
        self.l2_norm = F.normalize


    def mask_pooling(self, x, mask=None):
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.bool)
        masked_x = torch.mul(x, (~mask).unsqueeze(-1))
        sum_x = torch.sum(masked_x, dim=1)
        avg_x = sum_x / (torch.sum(~mask, dim=1).unsqueeze(-1))

        return avg_x


    def forward(self, x, padding_mask=None) -> Tensor:
        x_out = self.linear1(x)

        if padding_mask is not None:
            x_out = self.encoder(x_out, src_key_padding_mask=padding_mask)
        else:
            x_out = self.encoder(x_out)

        x_out = self.mask_pooling(x_out, padding_mask)
        x_out = self.linear2(x_out)
        x_out = self.l2_norm(x_out)

        return x_out


if __name__ == "__main__":
    model = MT(
        input_dim=128,
        output_dim=128,
        dim=256,
        depth=10,
        heads=8,
        dropout=0.1,
    ).cuda()

    src_padding_mask = torch.zeros((15000, 64), dtype = bool).cuda()
    x = torch.randn(15000, 64, 128).cuda()
    print(model(x, src_padding_mask).shape)
