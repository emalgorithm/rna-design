import torch.nn as nn
from src.lstm.lstm_model import LSTMModel
from src.data_util.data_constants import tag_to_ix, word_to_ix

import torch.nn as nn
import torch
import torch.nn.functional as F


class RNNGenerator(nn.Module):
    def __init__(self, device="cpu"):
        super(RNNGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size=len(tag_to_ix), hidden_size=256, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.num_directions = 2
        self.batch_size = 1
        self.num_layers = 1
        self.device = device
        self.hidden_dim = 256
        self.hidden = self.init_hidden()
        self.fc = nn.Linear(256 * 2, len(word_to_ix))

    def init_hidden(self, initial_hidden=None):
        if initial_hidden is None:
            initial_hidden = torch.zeros(self.hidden_dim).to(self.device)
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        return (initial_hidden.repeat(self.num_layers * self.num_directions, self.batch_size,
                                      1).to(self.device),
                torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, x, _, z):
        x = torch.unsqueeze(x, 0)
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        out = self.fc(lstm_out)
        base_prob = F.softmax(out, dim=2)

        return torch.squeeze(base_prob)



# class RNNGenerator(nn.Module):
#     def __init__(self):
#         super(RNNGenerator, self).__init__()
#
#         self.lstm_model = LSTMModel(embedding_dim=8, hidden_dim=128, num_layers=1,
#                                     vocab_size=len(tag_to_ix),
#                                     output_size=len(word_to_ix))
#
#     def forward(self, seq, seq_lengths, z):
#         base_scores = self.lstm_model(seq, seq_lengths, initial_hidden=z)
#
#         return base_scores
