import torch.nn as nn
import torch.nn.functional as F
from src.gcn.graph_convolution import GraphConvolution


class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim, n_classes, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        # TODO: Check softmax dimension
        return F.softmax(x, dim=2)
