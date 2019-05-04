import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, GINConv, GATConv
from src.gcn.graph_convolution import GraphConvolution


class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim, n_classes, n_conv_layers=3, dropout=0, device='cpu'):
        super(GCN, self).__init__()
        # self.conv1 = GCNConv(n_features, hidden_dim)
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.conv3 = GCNConv(hidden_dim, n_classes)

        # self.conv1 = GINConv(n_features, hidden_dim)
        # self.conv2 = GINConv(hidden_dim, hidden_dim)
        # self.conv3 = GINConv(hidden_dim, n_classes)
        self.convs = nn.ModuleList()

        # First layer
        # self.convs.append(GATConv(n_features, hidden_dim))
        nn1 = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim
                                                                           * n_features))
        self.convs.append(NNConv(n_features, hidden_dim, nn1))

        # Hidden layers
        for i in range(n_conv_layers - 1):
            nn2 = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,
                                                                               hidden_dim * hidden_dim))
            # self.convs.append(GATConv(hidden_dim, hidden_dim))
            self.convs.append(NNConv(hidden_dim, hidden_dim, nn2))

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, n_classes)

        # nn1 = nn.Linear(1, n_features)
        # self.conv1 = NNConv(n_features, hidden_dim, nn1)
        # nn2 = nn.Linear(hidden_dim, hidden_dim)
        # self.conv2 = NNConv(hidden_dim, hidden_dim, nn2)
        # nn3 = nn.Linear(hidden_dim, hidden_dim)
        # self.conv3 = NNConv(hidden_dim, n_classes, nn3)

        # self.conv1 = GraphConvolution(n_features, hidden_dim)
        # self.conv2 = GraphConvolution(hidden_dim, hidden_dim)
        # self.conv3 = GraphConvolution(hidden_dim, n_classes)

        self.dropout = dropout

    def forward(self, data):
        x, adj, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            # x = conv(x, adj)
            x = conv(x, adj, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.fc(x)
        # x = self.conv1(x, adj)
        # x = self.conv2(x, adj)
        # x = self.conv3(x, adj)

        return F.log_softmax(x, dim=1)
