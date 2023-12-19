import torch
from torch.nn import functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class MLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, bidirectional, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirctional = bidirectional
        self.dropout = dropout

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, output_dim)


    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)

        if self.lstm.bidirectional is False:
            hidden = hidden[-1]
            output = self.fc1(hidden)
        else:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            output = self.fc2(hidden)

        return output


if __name__ == "__main__":
    lstm = MLSTM(12, 24, 12, 2, True, 0.1).cuda()
    x = torch.randn(5, 10, 12).cuda()
    lengths = torch.tensor([10, 9, 8, 7, 6]).cuda()
    print(lstm(x, lengths).shape)