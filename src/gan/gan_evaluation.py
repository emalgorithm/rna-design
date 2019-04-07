import RNA
from src.data_util.data_processing import one_hot_embed_sequence, prepare_sequences, decode_sequence
from data_util.data_constants import word_to_ix, tag_to_ix, ix_to_word
from src.util import dotbracket_to_graph
from src.gcn.gcn_util import sparse_mx_to_torch_sparse_tensor
import networkx as nx
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import hamming_loss


def evaluate_gan(generator, data_loader, n_features, device):
    generator.eval()
    matches = []
    hamming_dists = []

    for i, (seq, dot_bracket) in enumerate(data_loader):
        # Train only on sequences with interesting fold
        if ')' not in dot_bracket[0]:
            continue

        ###
        # Extract input for all models
        ###
        # Batch contains a single element, extract it
        dot_bracket = dot_bracket[0]
        seq = seq[0]

        # For RNN Model
        targets, _ = prepare_sequences([seq], word_to_ix)
        targets.to(device)
        sequences, sequences_lengths = prepare_sequences([dot_bracket], tag_to_ix)
        sequences.to(device)
        sequences_lengths.to(device)

        g = dotbracket_to_graph(dot_bracket)
        # sample_y = nx.adjacency_matrix(g, nodelist=sorted(list(g.nodes())))
        # adj = sparse_mx_to_torch_sparse_tensor(sample_y).to(device)

        # x = one_hot_embed_sequence(seq, word_to_ix).to(device)
        hot_embedded_dot_bracket = one_hot_embed_sequence(dot_bracket, tag_to_ix).to(device)

        z = Variable(torch.FloatTensor(np.random.normal(0, 1, n_features)).to(device))

        # Generate graph features
        generated_x = generator(hot_embedded_dot_bracket, sequences_lengths, z)
        pred = generated_x.max(1)[1].cpu().numpy()
        pred_sequence = decode_sequence(pred, ix_to_word)

        pred_dot_bracket = RNA.fold(pred_sequence)[0]
        matches.append(pred_dot_bracket == dot_bracket)
        hamming_dists.append(hamming_loss(list(dot_bracket), list(pred_dot_bracket)))

    print("Accuracy: {0:.2f}".format(np.mean(matches)))
    print("Hamming loss: {0:.2f}".format(np.mean(hamming_dists)))
