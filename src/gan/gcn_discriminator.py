import torch.nn as nn
from src.gcn.gcn_graph_classification import GCNGraphClassification


class GCNDiscriminator(nn.Module):
    def __init__(self, n_features):
        super(GCNDiscriminator, self).__init__()
        self.gcn = GCNGraphClassification(n_features=n_features, hidden_dim=n_features,
                                          n_classes=1, dropout=0)

    def forward(self, x, adj):
        validity = self.gcn(x, adj)

        return validity
