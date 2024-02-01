import torch
from torch.nn import functional as F
from torchvision.models import resnet34, resnet18, resnet50, efficientnet_b0
from torch import nn
from torch import Tensor
from typing import Optional


class MCNN(nn.Module):
    def __init__(self, emd_size=128):
        super().__init__()
        self.emd_size = emd_size
        self.resnet = resnet50()
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
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.fc = nn.Linear(8192, self.emd_size)
        self.l2_norm = F.normalize


    def forward(self, x, padding_mask=None) -> Tensor:
        x_out = x.unsqueeze(1)
        x_out = self.conv(x_out)
        x_out = self.backbone(x_out)
        x_out = self.fc(x_out)
        x_out = self.l2_norm(x_out)

        return x_out


if __name__ == "__main__":
    model = MCNN()
    src_padding_mask = torch.tensor(
        [[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]], dtype=torch.bool
    )
    print(model(torch.rand(2, 5, 128), src_padding_mask).shape)
