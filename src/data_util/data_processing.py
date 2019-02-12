import numpy as np
import torch


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def prepare_sequences(seqs, to_ix):
    X = [[to_ix[word] for word in sentence] for sentence in seqs]
    X_lengths = [len(sentence) for sentence in X]

    # create an empty matrix with padding tokens
    pad_token = to_ix['<PAD>']
    longest_sent = max(X_lengths)
    batch_size = len(X)
    padded_X = np.ones((batch_size, longest_sent)) * pad_token
    # copy over the actual sequences
    for i, x_len in enumerate(X_lengths):
        sequence = X[i]
        padded_X[i, 0:x_len] = sequence[:x_len]

    return torch.tensor(padded_X, dtype=torch.long)
