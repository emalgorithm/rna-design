import os
import sys
sys.path.append(os.getcwd().split('src')[0])

from sklearn.metrics import classification_report, matthews_corrcoef
import pickle
import torch

from src.data_util.data_constants import families, word_to_ix
from src.data_util.rna_family_graph_dataset import RNAFamilyGraphDataset
from torch_geometric.data import DataLoader
from src.gcn.gcn import GCN

test_dataset = '../data/family_prediction/dataset_Rfam_validated_2400_12classes.fasta'
foldings_dataset = '../data/family_prediction/foldings.pkl'
model_name = "hidden_dim_100_dropout_0.2"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_classes = len(families)

test_set = RNAFamilyGraphDataset(test_dataset, foldings_dataset)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

opt = pickle.load(open('../results_family_classification/' + model_name +
                       '/hyperparams.pkl', "rb"))

# model = GCN(n_features=opt.embedding_dim, hidden_dim=opt.hidden_dim, n_classes=n_classes,
#             n_conv_layers=opt.n_conv_layers,
#             dropout=opt.dropout, batch_norm=opt.batch_norm, num_embeddings=len(word_to_ix),
#             embedding_dim=opt.embedding_dim,
#             node_classification=False).to(opt.device)
model = GCN(n_features=20, hidden_dim=opt.hidden_dim, n_classes=n_classes,
            n_conv_layers=opt.n_conv_layers,
            dropout=opt.dropout, batch_norm=opt.batch_norm, num_embeddings=len(word_to_ix),
            embedding_dim=20,
            node_classification=False).to(opt.device)
model.load_state_dict(torch.load('../models_family_classification/' + model_name + '/model.pt',
                                 map_location=device))

y_pred = []
y_true = []

for batch_idx, data in enumerate(test_loader):
    model.eval()

    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_attr = data.edge_attr.to(device)
    data.batch = data.batch.to(device)
    data.y = data.y.to(device)

    out = model(data)

    pred = out.max(1)[1]

    y_pred += list(pred.cpu().numpy())
    y_true += list(data.y.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=families, digits=4))
print("MCC: {0:.4f}".format(matthews_corrcoef(y_true, y_pred)))
