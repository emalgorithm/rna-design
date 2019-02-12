import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size=4, output_size=3, batch_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2base = nn.Linear(hidden_dim, output_size)
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, sentence):
        self.hidden = self.init_hidden()

        # sentence has shape (batch_size, seq_length)
        # embeds has shape (batch_size, seq_length, embedding_dim)
        embeds = self.word_embeddings(sentence)

        # lstm_in has shape (seq_length, batch_size, embedding_dim)
        lstm_in = embeds.permute(1, 0, 2)

        # lstm_in has shape (seq_length, batch_size, hidden_dim)
        lstm_out, self.hidden = self.lstm(lstm_in, self.hidden)

        # Flatten lstm_out to shape (seq_length * batch_size, hidden_dim) and apply linear layer to
        # all the output representation of the basis
        base_space = self.hidden2base(lstm_out.view(-1, lstm_out.size(2)))

        # base_scores has shape (seq_length * batch_size, output_size)
        base_scores = F.log_softmax(base_space, dim=1)
        return base_scores
