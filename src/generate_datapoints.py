import RNA
import pickle
from util import dotbracket_to_graph, get_family_to_sequences, get_all_sequences
import networkx as nx
import numpy as np

# individual_sequences_path = '../data/temp_train/'
# family_to_sequences = get_family_to_sequences()

# for family, sequences in family_to_sequences.items():
#     for sequence in sequences:
#         if len(sequence) == 30:
#             sequence_file = '{}.rna'.format(i)
#             i += 1
#             dot_bracket = RNA.fold(sequence)[0]
#             adj_matrix = nx.to_scipy_sparse_matrix(dotbracket_to_graph(dot_bracket))
#             print("Compute structure for sequence {}".format(i))
#             obj = {
#                 'sequence': sequence,
#                 'dot_bracket': dot_bracket,
#                 'family': family,
#                 'adj_matrix': adj_matrix,
#             }
#             pickle.dump(obj, open(individual_sequences_path + sequence_file, 'wb'))


# def store_sequences(sequences, path):
#     i = 1
#     for sequence in sequences:
#         sequence_file = '{}.rna'.format(i)
#         i += 1
#         dot_bracket = RNA.fold(sequence)[0]
#         adj_matrix = nx.to_scipy_sparse_matrix(dotbracket_to_graph(dot_bracket))
#         print("Compute structure for sequence {}".format(i))
#         obj = {
#             'sequence': sequence,
#             'dot_bracket': dot_bracket,
#             # 'family': family,
#             # 'adj_matrix': adj_matrix,
#         }
#         pickle.dump(obj, open(path + sequence_file, 'wb'))


# sequences = np.array([sequence for sequence in get_all_sequences() if len(sequence) <= 100])
# sequences = get_all_sequences()
# np.random.shuffle(sequences)
sequences_with_folding = []

family_to_sequences = get_family_to_sequences()

for family, sequences in list(family_to_sequences.items()):
    for sequence in sequences:
        dot_bracket = RNA.fold(sequence)[0]
        print("Compute structure for sequence {}".format(len(sequences_with_folding)))
        datapoint = {
            'sequence': sequence,
            'dot_bracket': dot_bracket,
            'family': family,
        }
        sequences_with_folding.append(datapoint)

np.random.shuffle(sequences_with_folding)
pickle.dump(sequences_with_folding, open('../data/sequences_with_folding.pkl', 'wb'))

# train, val, test = np.split(sequences, [int(.8*len(sequences)), int(.9*len(sequences))])
#
# store_sequences(train, '../data/less_than_100/train/')
# store_sequences(val, '../data/less_than_100/val/')
# store_sequences(test, '../data/less_than_100/test/')



