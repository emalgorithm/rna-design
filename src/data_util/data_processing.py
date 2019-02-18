import numpy as np
import torch
from data_util.data_constants import word_to_ix, tag_to_ix


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def prepare_sequences(seqs, to_ix):
    seqs.sort(key=lambda s: len(s), reverse=True)
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

    return torch.tensor(padded_X, dtype=torch.long), torch.tensor(X_lengths, dtype=torch.long)


def my_collate(batch):
    sequences, sequences_lengths = prepare_sequences([item[0] for item in batch], word_to_ix)
    targets, _ = prepare_sequences([item[1] for item in batch], tag_to_ix)

    return [sequences, targets, sequences_lengths]


