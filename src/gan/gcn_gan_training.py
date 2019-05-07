import os
import sys
sys.path.append(os.getcwd().split('src')[0])

import argparse
import numpy as np
import pickle
import time

from torch_geometric.data import DataLoader

import torch.nn as nn
import torch

from src.gcn.gcn import GCN
from src.data_util.rna_graph_dataset import RNAGraphDataset
from src.data_util.data_constants import word_to_ix, tag_to_ix
from src.evaluation import compute_metrics_graph, evaluate_struct_to_seq_graph
from src.gan.gan import GAN
from src.gan.wgan import WGAN
from src.visualization_util import plot_loss

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
parser.add_argument('--gan_type', type=str, default="gan", help='Which type of gan to use')
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

probability = not opt.gan_type == "wgan"
random_features = 5
n_features = 1 + random_features
n_classes = len(word_to_ix)
# Initialize generator and discriminator
generator = GCN(n_features, hidden_dim=opt.hidden_dim, n_classes=n_classes, n_conv_layers=opt.n_conv_layers,
                dropout=opt.dropout, softmax=True).to(opt.device)
discriminator = GCN(len(word_to_ix), hidden_dim=opt.hidden_dim, n_classes=1,
                    n_conv_layers=opt.n_conv_layers, dropout=opt.dropout,
                    node_classification=False, probability=probability).to(opt.device)

if opt.gan_type == "gan":
    gan = GAN(generator, discriminator, opt.learning_rate, opt.device)
elif opt.gan_type == "wgan":
    gan = WGAN(generator, discriminator, device=opt.device)


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


def train_epoch(epoch_idx):
    generator.train()
    discriminator.train()
    for batch_idx, data in enumerate(train_loader):
        data.x = data.x.to(opt.device)
        data.edge_index = data.edge_index.to(opt.device)
        data.edge_attr = data.edge_attr.to(opt.device)
        data.batch = data.batch.to(opt.device)
        data.sequence = data.sequence.to(opt.device)
        dot_bracket = data.y.to(opt.device)

        # Sample noise as generator input
        z = torch.Tensor(np.random.normal(0, 1, (data.x.shape[0], random_features))).to(opt.device)
        data.x = torch.cat((data.x, z), dim=1)

        g_loss, d_loss = gan.train(data)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch_idx, opt.n_epochs, batch_idx, len(train_loader), d_loss.item(), g_loss.item())
        )

    return 0, 0, 0


def run(n_epochs, results_dir, model_dir):
    # print("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if opt.evaluate_training:
        loss, h_loss, accuracy = evaluate_struct_to_seq_graph(generator, train_loader,
                                                              gan=True,
                                                              n_random_features=random_features,
                                                              batch_size=opt.batch_size,
                                                              mode='val',
                                                              device=opt.device,
                                                              verbose=opt.verbose)
        train_losses = [loss]
        train_h_losses = [h_loss]
        train_accuracies = [accuracy]
    else:
        train_losses = []
        train_h_losses = []
        train_accuracies = []

    val_loss, val_h_loss, val_accuracy = evaluate_struct_to_seq_graph(generator, val_loader,
                                                                      gan=True,
                                                                      n_random_features=random_features,
                                                                      batch_size=opt.batch_size,
                                                                      mode='val',
                                                                      device=opt.device,
                                                                      verbose=opt.verbose)

    # test_losses = []
    val_losses = [val_loss]
    # test_h_losses = []
    val_h_losses = [val_h_loss]
    # test_accuracies = []
    val_accuracies = [val_accuracy]

    for epoch in range(n_epochs):
        start = time.time()
        print("Epoch {}: ".format(epoch + 1))

        loss, h_loss, accuracy = train_epoch(epoch)
        # test_loss, test_h_loss, test_accuracy = evaluate_struct_to_seq_graph(model, test_loader,
        #                                                                      loss_function, mode='test', device=opt.device)
        val_loss, val_h_loss, val_accuracy = evaluate_struct_to_seq_graph(generator, val_loader,
                                                                          gan=True,
                                                                          n_random_features=random_features,
                                                                          batch_size=opt.batch_size,
                                                                          mode='val',
                                                                          device=opt.device,
                                                                          verbose=opt.verbose)
        end = time.time()
        print("Epoch took {0:.2f} seconds".format(end - start))

        # if not val_h_losses or val_h_loss > max(val_h_losses):
        #     torch.save(model.state_dict(), model_dir + 'model.pt')
        #     print("Saved updated model")

        train_losses.append(loss)
        # test_losses.append(test_loss)
        val_losses.append(val_loss)
        train_h_losses.append(h_loss)
        # test_h_losses.append(test_h_loss)
        val_h_losses.append(val_h_loss)
        train_accuracies.append(accuracy)
        # test_accuracies.append(test_accuracy)
        val_accuracies.append(val_accuracy)

        plot_loss(train_losses, val_losses, file_name=results_dir + 'loss.jpg')
        plot_loss(train_h_losses, val_h_losses, file_name=results_dir + 'h_loss.jpg',
                  y_label='hamming_loss')
        plot_loss(train_accuracies, val_accuracies, file_name=results_dir + 'acc.jpg',
                  y_label='accuracy')

        pickle.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            # 'test_losses': test_losses,
            'train_h_losses': train_h_losses,
            'val_h_losses': val_h_losses,
            # 'test_h_losses': test_h_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            # 'test_accuracies': test_accuracies
        }, open(results_dir + 'scores.pkl', 'wb'))

        # if len(val_h_losses) > opt.early_stopping and min(val_h_losses[-opt.early_stopping:]) > \
        #         min(val_h_losses):
        #     print("Training terminated because of early stopping")
        #     print("Best val_loss: {}".format(min(val_losses)))
        #     print("Best val_h_loss: {}".format(min(val_h_losses)))
        #     print("Best val_accuracy: {}".format(max(val_accuracies)))
        #     break


def main():
    results_dir = '../results/{}/'.format(opt.model_name)
    model_dir = '../models/{}/'.format(opt.model_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(results_dir + 'hyperparams.txt', 'w') as f:
        f.write(str(opt))

    with open(results_dir + 'hyperparams.pkl', 'wb') as f:
        pickle.dump(opt, f)

    run(opt.n_epochs, results_dir, model_dir)


if __name__ == "__main__":
    main()
