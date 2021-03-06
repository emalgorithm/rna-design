import os
import sys
sys.path.append(os.getcwd().split('src')[0])

from data_util.data_processing import prepare_sequence, my_collate_seq_to_struct
from lstm.lstm_model import LSTMModel
import torch
import torch.nn as nn
import torch.optim as optim
from data_util.rna_dataset import RNADataset
from data_util.rna_dataset_single_file import RNADatasetSingleFile
from torchvision import transforms
from visualization_util import plot_loss
from data_util.data_constants import word_to_ix, tag_to_ix
from evaluation import masked_hamming_loss, compute_accuracy, evaluate
import pickle
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="test10", help='model name')
parser.add_argument('--device', default="cpu", help='cpu or cuda')
parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to train on')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of samples to train on')
parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden '
                                                                'representations of LSTM')
parser.add_argument('--embedding_dim', type=int, default=6, help='Dimension of embedding for '
                                                                   'the bases')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--seq_max_len', type=int, default=100, help='Maximum length of sequences '
                                                                 'used for training and testing')
parser.add_argument('--lstm_layers', type=int, default=2, help='Number of layers of the lstm')
parser.add_argument('--dropout', type=float, default=0, help='Amount of dropout')

opt = parser.parse_args()
print(opt)

model = LSTMModel(opt.embedding_dim, opt.hidden_dim, num_layers=opt.lstm_layers, vocab_size=len(word_to_ix),
                  output_size=len(tag_to_ix), batch_size=opt.batch_size, device=opt.device)
loss_function = nn.NLLLoss(ignore_index=tag_to_ix['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

# Data Loading
x_transform = transforms.Lambda(lambda sequences: prepare_sequence(sequences, word_to_ix))
y_transform = transforms.Lambda(lambda sequences: prepare_sequence(sequences, tag_to_ix))

# train_set = RNADataset('../../data/temp_train/', x_transform=x_transform, y_transform=y_transform)
# test_set = RNADataset('../../data/temp_test/', x_transform=x_transform, y_transform=y_transform)
# train_set = RNADataset('../data/less_than_40/train/')
# test_set = RNADataset('../data/less_than_40/test/')
# val_set = RNADataset('../data/less_than_40/val/')
# train_set = RNADataset('../data/less_than_450/train/')
# test_set = RNADataset('../data/less_than_450/test/')
# val_set = RNADataset('../data/less_than_450/val/')
n_train_samples = None if not opt.n_samples else int(opt.n_samples * 0.8)
n_val_samples = None if not opt.n_samples else int(opt.n_samples * 0.1)
train_set = RNADatasetSingleFile('../data/sequences_with_folding_train.pkl',
                                 seq_max_len=opt.seq_max_len, n_samples=n_train_samples)
test_set = RNADatasetSingleFile('../data/sequences_with_folding_test.pkl',
                                seq_max_len=opt.seq_max_len, n_samples=n_val_samples)
val_set = RNADatasetSingleFile('../data/sequences_with_folding_val.pkl',
                               seq_max_len=opt.seq_max_len, n_samples=n_val_samples)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,
                                           collate_fn=my_collate_seq_to_struct)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False,
                                          collate_fn=my_collate_seq_to_struct)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size, shuffle=False,
                                         collate_fn=my_collate_seq_to_struct)


def train_epoch(model, train_loader):
    model.train()
    avg_loss = 0
    h_loss = 0
    accuracy = 0

    for batch_idx, (sequences, dot_brackets, sequences_lengths) in enumerate(train_loader):
        sequences = sequences.to(opt.device)
        dot_brackets = dot_brackets.to(opt.device)
        sequences_lengths = sequences_lengths.to(opt.device)

        # Skip last batch if it does not have full size
        if sequences.shape[0] < opt.batch_size:
            continue
        model.zero_grad()

        base_scores = model(sequences, sequences_lengths)

        loss = loss_function(base_scores, dot_brackets.view(-1))
        avg_loss += loss
        loss.backward()
        optimizer.step()

        pred = base_scores.max(1)[1]
        h_loss += masked_hamming_loss(dot_brackets.view(-1).cpu().numpy(), pred.cpu().numpy())
        accuracy += compute_accuracy(dot_brackets.cpu().numpy(), pred.view_as(
            dot_brackets).cpu().numpy())

    avg_loss /= len(train_loader)
    h_loss /= len(train_loader)
    accuracy /= len(train_loader)

    print("training loss is {}".format(avg_loss))
    print("training hamming loss: {}".format(h_loss))
    print("accuracy: {}".format(accuracy))

    return avg_loss.item(), h_loss, accuracy


def run(model, n_epochs, train_loader, test_loader, results_dir, model_dir):
    print("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    train_losses = []
    # test_losses = []
    val_losses = []
    train_h_losses = []
    # test_h_losses = []
    val_h_losses = []
    train_accuracies = []
    # test_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        start = time.time()
        print("Epoch {}: ".format(epoch + 1))

        loss, h_loss, accuracy = train_epoch(model, train_loader)
        # test_loss, test_h_loss, test_accuracy = evaluate(model, test_loader, loss_function,
        #                                                  batch_size, mode='test', device=opt.device)
        val_loss, val_h_loss, val_accuracy = evaluate(model, val_loader, loss_function,
                                                      opt.batch_size, mode='val', device=opt.device)
        end = time.time()
        print("Epoch took {0:.2f} seconds".format(end - start))

        if not val_accuracies or val_accuracy > max(val_accuracies):
            torch.save(model.state_dict(), model_dir + 'model.pt')
            print("Saved updated model")

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


def main():
    results_dir = '../results/{}/'.format(opt.model_name)
    model_dir = '../models/{}/'.format(opt.model_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(results_dir + 'hyperparams.txt', 'w') as f:
        f.write(str(opt))

    run(model, opt.n_epochs, train_loader, test_loader, results_dir, model_dir)


if __name__ == "__main__":
    main()

