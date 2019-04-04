import torch.nn as nn
from src.gcn.gcn import GCN


class GCNGenerator(nn.Module):
    def __init__(self, n_features):
        """
        :param n_features: Number of features for the nodes.
        """
        super(GCNGenerator, self).__init__()

        self.gcn = GCN(n_features=n_features, hidden_dim=n_features, n_classes=6, dropout=0)

    def forward(self, adj, z, n_nodes):
        """
        Given an RNA graph with no features, it assigns a base (A, U, C, G) to each node as a one hot embedding.
        :param adj: adjacency matrix of the RNA graph
        :param z: GAN noise input
        :param n_nodes: number of nodes in the graph
        :return: feature matrix of shape [n_nodes, n_features] where the features is a one-hot
        embedding representing the base assigned to that node
        """
        x = z.repeat(n_nodes, 1)
        sequence = self.gcn(x, adj)

        return sequence
