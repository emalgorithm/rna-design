from torch_geometric.data import InMemoryDataset, Data
import pickle
import numpy as np
import torch
from sklearn.preprocessing import scale

from src.data_util.data_processing import prepare_sequence
from src.data_util.data_constants import word_to_ix, tag_to_ix
from src.util import dotbracket_to_graph


class RNAGraphDataset(InMemoryDataset):
    def __init__(self, file_path, transform=None, pre_transform=None, seq_max_len=40, seq_min_len=1,
                 n_samples=None):
        super(RNAGraphDataset, self).__init__(file_path, transform, pre_transform)

        self.data = pickle.load(open(file_path, 'rb'))
        self.data = [x for x in self.data if seq_min_len <= len(x['sequence']) <= seq_max_len]
        self.data = self.data if not n_samples else self.data[:n_samples]
        lengths = [len(x['sequence']) for x in self.data]

        print("{} sequences found at path {} with max length {}, average length of {}, "
              "and median length of {}".format(len(self.data), file_path, seq_max_len,
                                               np.mean(lengths), np.median(lengths)))

        data_list = []

        for x in self.data:
            sequence_string = x['sequence']
            sequence = prepare_sequence(sequence_string, word_to_ix)

            dot_bracket_string = x['dot_bracket']
            dot_bracket = prepare_sequence(dot_bracket_string, tag_to_ix)

            g = dotbracket_to_graph(dot_bracket_string)

            degrees = [g.degree[i] for i in range(len(g))]
            # Standardize features
            degrees = scale(degrees)
            x = torch.Tensor([degrees]).t().contiguous()

            edges = list(g.edges(data=True))
            # One-hot encoding of the edge type
            edge_attr = torch.Tensor([[0, 1] if e[2]['edge_type'] == 'adjacent' else [1, 0] for e in
                                      edges])
            edge_index = torch.LongTensor(list(g.edges())).t().contiguous()
            # y = torch.cat((sequence.unsqueeze(0), dot_bracket.unsqueeze(0)), 0)
            y = dot_bracket

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, sequence=sequence)
            # data.sequence_string = sequence_string
            # data.sequence = sequence
            # data.dot_bracket_string = dot_bracket_string
            # data.dot_bracket = dot_bracket

            data_list.append(data)

        self.data, self.slices = self.collate(data_list)

    def download(self):
        pass

    def process(self):
        pass

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []
