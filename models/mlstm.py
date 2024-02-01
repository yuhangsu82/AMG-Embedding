import torch
from torch.nn import functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch import Tensor
from typing import Optional


class MLSTM(nn.Module):
    def __init__(
        self,
        input_dim=128,
        output_dim=128,
        hidden_dim=256,
        num_layers=6,
        dropout=0.1,
    ):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, dropout=self.dropout, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        self.l2_norm = F.normalize


    def forward(self, x, padding_mask=None) -> Tensor:
        lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.int64).to("cpu")
        lengths = lengths - padding_mask.sum(dim=1).to("cpu")

        packed_input = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_input)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.shape[1]
        )

        last_seq_index = lengths.to(x.device) - 1
        last_output = output[torch.arange(x.shape[0]), last_seq_index]
        x_out = self.fc(last_output)
        x_out = self.l2_norm(x_out)

        return x_out


if __name__ == "__main__":
    lstm = MLSTM(128, 128, 256, 2, 0.1).cuda()
    x = torch.randn(2, 5, 128).cuda()
    src_padding_mask = torch.tensor(
        [[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]], dtype=torch.bool
    )
    print(lstm(x, src_padding_mask).shape)
