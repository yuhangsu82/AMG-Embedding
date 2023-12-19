import torch
from torch.nn import functional as F
from torchvision.models import resnet34
from torch import nn

class MCNN(nn.Module):
    def __init__(self, emd_size=256):
        super().__init__()
        self.emd_size = emd_size
        self.resnet = resnet34()
        # self.resnet = resnet18()
        self.musicNet = nn.Sequential(
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

        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.fc = nn.Linear(4096, emd_size)
        self.l2_norm = F.normalize

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.musicNet(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = MCNN()
    print(model(torch.ones(2, 64, 128)))