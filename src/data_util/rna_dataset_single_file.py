from torch.utils.data import Dataset
import pickle
import numpy as np
from src.util import dotbracket_to_graph
import networkx as nx
from src.gcn.gcn_util import sparse_mx_to_torch_sparse_tensor


class RNADatasetSingleFile(Dataset):
    def __init__(self, file_path, x_transform=None, y_transform=None, seq_max_len=40, seq_min_len=1,
                 n_samples=None, graph=False):
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.file_path = file_path
        self.graph = graph

        self.data = pickle.load(open(file_path, 'rb'))
        self.data = [x for x in self.data if seq_min_len <= len(x['sequence']) <= seq_max_len]
        self.data = self.data if not n_samples else self.data[:n_samples]
        lengths = [len(x['sequence']) for x in self.data]

        print("{} sequences found at path {} with max length {}, average length of {}, "
              "and median length of {}".format(len(self.data), file_path, seq_max_len,
                                               np.mean(lengths), np.median(lengths)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample_x = sample['sequence']
        sample_y = sample['dot_bracket']

        if self.x_transform:
            sample_x = self.x_transform(sample_x)

        if self.y_transform:
            sample_y = self.y_transform(sample_y)

        if self.graph:
            # Convert dot_bracket string to graph
            g = dotbracket_to_graph(sample_y)
            sample_y = nx.adjacency_matrix(g, nodelist=sorted(list(g.nodes())))
            sample_y = sparse_mx_to_torch_sparse_tensor(sample_y)

        return sample_x, sample_y
