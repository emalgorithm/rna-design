import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data_util.data_constants import word_to_ix

torch.manual_seed(1)


class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers=2, vocab_size=4, output_size=3,
                 batch_size=1, bidirectional=True, device="cpu", dropout=0):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        padding_idx = word_to_ix['<PAD>']
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx).to(device)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True,
                            batch_first=True, dropout=dropout).to(device)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2base = nn.Linear(self.num_directions * hidden_dim, output_size).to(
            device)
        self.nn_dropout = nn.Dropout(dropout)
        self.batch_size = batch_size
        self.device = device
        self.hidden = self.init_hidden()

    def init_hidden(self, initial_hidden=None):
        if initial_hidden is None:
            initial_hidden = torch.zeros(self.hidden_dim).to(self.device)
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        return (initial_hidden.repeat(self.num_layers * self.num_directions, self.batch_size,
                                      1).to(self.device),
                torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device))
        # return (torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device),
        #         torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, sentence, sentence_lenghts, initial_hidden=None):
        self.hidden = self.init_hidden(initial_hidden)
        # sentence has shape (batch_size, seq_length)
        # embeds has shape (batch_size, seq_length, embedding_dim)
        embeds = self.word_embeddings(sentence)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        embeds = nn.utils.rnn.pack_padded_sequence(embeds, sentence_lenghts, batch_first=True)

        # lstm_out has shape (batch_size, seq_length, hidden_dim)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        # undo the packing operation
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Flatten lstm_out to shape (seq_length * batch_size, hidden_dim) and apply linear layer to
        # all the output representation of the basis
        # base_space has shape (seq_length * batch_size, output_size)
        # out = lstm_out.contiguous().view(-1, lstm_out.size(2))
        out = lstm_out
        out = self.nn_dropout(out)
        base_space = self.hidden2base(out)

        # base_scores has shape (seq_length * batch_size, output_size)
        base_scores = F.log_softmax(base_space, dim=2)
        return base_scores
