import os
import sys
sys.path.append(os.getcwd().split('src')[0])

import argparse
import numpy as np

from torch_geometric.data import DataLoader

import torch.nn as nn
import torch

from src.gcn.gcn import GCN
from src.data_util.rna_graph_dataset import RNAGraphDataset
from src.data_util.data_constants import word_to_ix, tag_to_ix
from src.evaluation import compute_metrics_graph, evaluate_struct_to_seq_graph

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument('--model_name', default="test10", help='model name')
parser.add_argument('--device', default="cpu", help='cpu or cuda')
parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to train on')
parser.add_argument('--n_epochs', type=int, default=10000, help='Number of samples to train on')
parser.add_argument('--hidden_dim', type=int, default=10, help='Dimension of hidden '
                                                                'representations of convolutional layers')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
parser.add_argument('--seq_max_len', type=int, default=100, help='Maximum length of sequences '
                                                                 'used for training and testing')
parser.add_argument('--seq_min_len', type=int, default=1, help='Maximum length of sequences '
                                                                 'used for training and testing')
parser.add_argument('--n_conv_layers', type=int, default=3, help='Number of convolutional layers')
parser.add_argument('--conv_type', type=str, default="MPNN", help='Type of convolutional layers')
parser.add_argument('--dropout', type=float, default=0, help='Amount of dropout')
parser.add_argument('--early_stopping', type=int, default=30, help='Number of epochs for early '
                                                                   'stopping')
parser.add_argument('--verbose', type=bool, default=False, help='Verbosity')
parser.add_argument('--evaluate_training', type=bool, default=False, help='Wheter to evaluate '
                                                                          'training results each '
                                                                          'epoch')
parser.add_argument('--train_dataset', type=str,
                    default='../data/folding_train.pkl', help='Path to training dataset')
parser.add_argument('--val_dataset', type=str,
                    default='../data/folding_val.pkl', help='Path to val dataset')
parser.add_argument('--test_dataset', type=str,
                    default='../data/folding_test.pkl', help='Path to test dataset')

opt = parser.parse_args()
print(opt)

# Loss function
adversarial_loss = torch.nn.BCELoss()

random_features = 5
n_features = 1 + random_features
n_classes = len(word_to_ix)
# Initialize generator and discriminator
generator = GCN(n_features, hidden_dim=opt.hidden_dim, n_classes=n_classes, n_conv_layers=opt.n_conv_layers,
                dropout=opt.dropout).to(opt.device)
discriminator = GCN(len(word_to_ix), hidden_dim=opt.hidden_dim, n_classes=1,
                    n_conv_layers=opt.n_conv_layers, dropout=opt.dropout,
                    node_classification=False).to(opt.device)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.learning_rate, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.learning_rate, betas=(opt.b1, opt.b2))

# Data Loading
n_train_samples = None if not opt.n_samples else int(opt.n_samples * 0.8)
n_val_samples = None if not opt.n_samples else int(opt.n_samples * 0.1)
train_set = RNAGraphDataset(opt.train_dataset, seq_max_len=opt.seq_max_len,
                            seq_min_len=opt.seq_min_len,
                            n_samples=n_train_samples)
# test_set = RNADatasetSingleFile(opt.test_dataset,
#                                 seq_max_len=opt.seq_max_len, seq_min_len=opt.seq_min_len,
#                                 n_samples=n_val_samples)
val_set = RNAGraphDataset(opt.val_dataset, seq_max_len=opt.seq_max_len, seq_min_len=opt.seq_min_len,
                          n_samples=n_val_samples)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
# # test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False,
# #                                           collate_fn=my_collate_struct_to_seq)
val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False)

# ----------
#  Training
# ----------

val_loss, val_h_loss, val_accuracy = evaluate_struct_to_seq_graph(generator, val_loader, gan=True,
                                                                  n_random_features=random_features,
                                                                  batch_size=opt.batch_size,
                                                                  mode='val',
                                                                  device=opt.device,
                                                                  verbose=opt.verbose)

val_losses = [val_loss]
val_h_losses = [val_h_loss]
val_accuracies = [val_accuracy]

for epoch in range(opt.n_epochs):
    generator.train()
    discriminator.train()
    for batch_idx, data in enumerate(train_loader):
        data.x = data.x.to(opt.device)
        data.edge_index = data.edge_index.to(opt.device)
        data.edge_attr = data.edge_attr.to(opt.device)
        data.batch = data.batch.to(opt.device)
        dot_bracket = data.y.to(opt.device)
        sequence = data.sequence.to(opt.device)
        n_graphs = torch.unique(data.batch).shape[0]

        # Adversarial ground truths
        valid = torch.ones([n_graphs, 1]).to(opt.device)
        fake = torch.zeros([n_graphs, 1]).to(opt.device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.Tensor(np.random.normal(0, 1, (data.x.shape[0], random_features))).to(opt.device)
        data.x = torch.cat((data.x, z), dim=1)

        # Generate a batch of images
        gen_sequences = generator(data)

        discriminator_data = data.clone()
        discriminator_data.x = gen_sequences
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(discriminator_data), valid)

        g_loss.backward(retain_graph=True)
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(discriminator_data), valid)
        fake_loss = adversarial_loss(discriminator(discriminator_data), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, batch_idx, len(train_loader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(train_loader) + batch_idx

    val_loss, val_h_loss, val_accuracy = evaluate_struct_to_seq_graph(generator, val_loader,
                                                                      gan=True,
                                                                      n_random_features=random_features,
                                                                      batch_size=opt.batch_size,
                                                                      mode='val',
                                                                      device=opt.device,
                                                                      verbose=opt.verbose)

    val_losses.append(val_loss)
    val_h_losses.append(val_h_loss)
    val_accuracies.append(val_accuracy)
