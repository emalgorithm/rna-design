import torch.nn as nn
import torch.nn.functional as F
from src.gcn.graph_convolution import GraphConvolution
import torch


class GCNGraphClassification(nn.Module):
    def __init__(self, n_features, hidden_dim, n_classes, dropout):
        super(GCNGraphClassification, self).__init__()

        self.gc1 = GraphConvolution(n_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # Take sum of node representation to get graph representation
        g = x.sum(dim=0)
        g = self.fc(g)
        return torch.sigmoid(g)
