import torch
from torch.nn import functional as F
from torchvision.models import resnet34, resnet18
from torch import nn
from torch import Tensor
from typing import Optional


class MCNN(nn.Module):
    def __init__(self, emd_size=128, is_residual=False):
        super().__init__()
        self.emd_size = emd_size
        self.resnet = resnet18()
        self.backbone = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            nn.Flatten(),
        )

        self.emd_size = emd_size
        self.is_residual = is_residual
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.fc = nn.Linear(2048, self.emd_size)
        self.l2_norm = F.normalize


    def mask_pooling(self, x, mask=None):
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.bool)
        masked_x = torch.mul(x, (~mask).unsqueeze(-1))
        sum_x = torch.sum(masked_x, dim=1)
        avg_x = sum_x / (torch.sum(~mask, dim=1).unsqueeze(-1))

        return avg_x


    def forward(self, x, padding_mask=None) -> Tensor:
        x_out = x.unsqueeze(1)
        x_out = self.conv(x_out)
        x_out = self.backbone(x_out)
        x_out = self.fc(x_out)
        if self.is_residual == True:
            if self.input_dim != self.output_dim:
                raise Exception("Residual can only used when input_dim == output_dim!")
            x_out = x_out + self.mask_pooling(x, padding_mask)
        x_out = self.l2_norm(x_out)

        return x_out


if __name__ == "__main__":
    model = MCNN()
    src_padding_mask = torch.tensor(
        [[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]], dtype=torch.bool
    )
    print(model(torch.rand(2, 5, 128), src_padding_mask).shape)
