import torch
import torch.nn.functional as F
from torch import nn

class TripletLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.loss_name = 'Triplet'
        self.alpha = alpha
        self.cosine_similarity = nn.CosineSimilarity()


    def forward(self, a_x, p_x, n_x):
        s_d = self.cosine_similarity(a_x, p_x)
        n_d = self.cosine_similarity(a_x, n_x)
        loss = torch.clamp(n_d - s_d + self.alpha, min=0.0)
        loss = torch.mean(loss)
        return loss
    

if __name__ == "__main__":
    loss_fc = TripletLoss(0.5)
    a = torch.randn((64, 128))
    b = torch.randn((64, 128))
    c = torch.randn((64, 128))
    print(loss_fc(a, b, c))