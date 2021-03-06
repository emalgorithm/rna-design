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
#         # adj_matrix = nx.to_scipy_sparse_matrix(dotbracket_to_graph(dot_bracket))
#         print("Compute structure for sequence {}".format(i))
#         obj = {
#             'sequence': sequence,
#             'dot_bracket': dot_bracket,
#             # 'family': family,
#             # 'adj_matrix': adj_matrix,
#         }
#         pickle.dump(obj, open(path + sequence_file, 'wb'))

# def store_sequences(sequences, path):
#     i = 1
#     for sequence in sequences:
#         sequence_file = '{}.rna'.format(i)
#         pickle.dump(sequence, open(path + sequence_file, 'wb'))
#         i += 1
#         if i % 10000 == 0:
#             print("Done {} sequences".format(i))


# sequences_with_folding = []
#
# family_to_sequences = get_family_to_sequences()
# i = 0
#
# for family, sequences in list(family_to_sequences.items()):
#     for sequence in sequences:
#         if len(sequence) <= 450:
#             dot_bracket = RNA.fold(sequence)[0]
#
#             if i % 1000 == 0:
#                 print("Compute structure for sequence {}".format(i))
#             i += 1
#
#             datapoint = {
#                 'sequence': sequence,
#                 'dot_bracket': dot_bracket,
#                 'family': family,
#             }
#             sequences_with_folding.append(datapoint)
#
# np.random.shuffle(sequences_with_folding)
# pickle.dump(sequences_with_folding, open('../data/sequences_with_folding.pkl', 'wb'))

# sequences = np.array([sequence for sequence in get_all_sequences() if len(sequence) <= 100])
# sequences = get_all_sequences()
sequences = pickle.load(open('../data/sequences_with_folding.pkl', 'rb'))
np.random.shuffle(sequences)


train, val, test = np.split(sequences, [int(.8*len(sequences)), int(.9*len(sequences))])

pickle.dump(train, open('../data/sequences_with_folding_train.pkl', 'wb'))
pickle.dump(val, open('../data/sequences_with_folding_val.pkl', 'wb'))
pickle.dump(test, open('../data/sequences_with_folding_test.pkl', 'wb'))

# store_sequences(train, '../data/less_than_450/train/')
# store_sequences(val, '../data/less_than_450/val/')
# store_sequences(test, '../data/less_than_450/test/')



