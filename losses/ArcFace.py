import torch
import torch.nn.functional as F
from torch import nn
import math


class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.loss_name = 'Arcface'
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin

    def forward(self, input, label, margin):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-5)
        cos_m = torch.cos(margin)
        sin_m = torch.sin(margin)
        th = torch.cos(math.pi - margin)
        mm = torch.sin(math.pi - margin) * margin
        phi = cosine * cos_m.unsqueeze(1) - sine * sin_m.unsqueeze(1)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > th.unsqueeze(1), phi, cosine - mm.unsqueeze(1))
        
        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        loss = nn.CrossEntropyLoss()(logits, label)

        return logits, loss


if __name__ == "__main__":
    arcface = ArcFaceLoss(in_features=128, out_features=10000, m=0.5, s=64.0).cuda()
    label = torch.tensor([1, 12, 334, 21]).cuda()
    print(label.shape)
    margin = torch.tensor([1, 0.5, 1, 1]).cuda()
    tensor = torch.randn(4,128).cuda()
    loss = arcface(tensor, label, margin)
    print(loss)