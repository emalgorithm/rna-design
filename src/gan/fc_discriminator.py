import torch.nn as nn
import torch


class FCDiscriminator(nn.Module):
    def __init__(self, n_features):
        super(FCDiscriminator, self).__init__()
        self.fc = nn.Linear(n_features, 1)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        out = self.fc2(x)
        validity = torch.sigmoid(out.mean(0))

        # out = self.fc(x.t())
        # validity = torch.sigmoid(out.mean(0))
        # out = self.relu(out)
        # out = self.fc2(out.t())
        # validity = torch.sigmoid(out)

        return validity
