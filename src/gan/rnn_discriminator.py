import torch.nn as nn
import torch


class RNNDiscriminator(nn.Module):
    def __init__(self, device="cpu"):
        super(RNNDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_size=9, hidden_size=256, num_layers=1,
                            bidirectional=False, batch_first=True)
        self.num_directions = 1
        self.batch_size = 1
        self.num_layers = 1
        self.hidden_dim = 256
        self.device = device
        self.hidden = self.init_hidden()
        self.fc = nn.Linear(256, 1)

    def init_hidden(self):
        return (torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, x, hot_embedded_dot_bracket):
        x = torch.cat([x, hot_embedded_dot_bracket], dim=1)
        x = torch.unsqueeze(x, 0)
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        out = self.fc(self.hidden[0])
        validity = torch.sigmoid(out)

        return validity
