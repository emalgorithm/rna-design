import os
import sys
sys.path.append(os.getcwd().split('src')[0])

from src.data_util.data_processing import prepare_sequence, my_collate_seq_to_struct, my_collate_struct_to_seq
from src.lstm.lstm_model import LSTMModel
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_util.rna_dataset import RNADataset
from src.data_util.rna_dataset_single_file import RNADatasetSingleFile
from torchvision import transforms
from src.visualization_util import plot_loss
from src.data_util.data_constants import word_to_ix, tag_to_ix
from src.evaluation import masked_hamming_loss, compute_accuracy, evaluate, \
    evaluate_struct_to_seq, compute_metrics
import pickle
import os
import time
import argparse
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="test10", help='model name')
parser.add_argument('--device', default="cpu", help='cpu or cuda')
parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to train on')
parser.add_argument('--n_epochs', type=int, default=10000, help='Number of samples to train on')
parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden '
                                                                'representations of LSTM')
parser.add_argument('--embedding_dim', type=int, default=6, help='Dimension of embedding for '
                                                                   'the bases')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.0004, help='Learning rate')
parser.add_argument('--seq_max_len', type=int, default=100, help='Maximum length of sequences '
                                                                 'used for training and testing')
parser.add_argument('--seq_min_len', type=int, default=1, help='Maximum length of sequences '
                                                                 'used for training and testing')
parser.add_argument('--lstm_layers', type=int, default=2, help='Number of layers of the lstm')
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

model = LSTMModel(opt.embedding_dim, opt.hidden_dim, num_layers=opt.lstm_layers, vocab_size=len(tag_to_ix),
                  output_size=len(word_to_ix), batch_size=opt.batch_size, device=opt.device)
loss_function = nn.NLLLoss(ignore_index=word_to_ix['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

# Data Loading
x_transform = transforms.Lambda(lambda sequences: prepare_sequence(sequences, tag_to_ix))
y_transform = transforms.Lambda(lambda sequences: prepare_sequence(sequences, word_to_ix))

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
train_set = RNADatasetSingleFile(opt.train_dataset, seq_max_len=opt.seq_max_len,
                                 seq_min_len=opt.seq_min_len,
                                 n_samples=n_train_samples)
# test_set = RNADatasetSingleFile(opt.test_dataset,
#                                 seq_max_len=opt.seq_max_len, seq_min_len=opt.seq_min_len,
#                                 n_samples=n_val_samples)
val_set = RNADatasetSingleFile(opt.val_dataset,
                               seq_max_len=opt.seq_max_len, seq_min_len=opt.seq_min_len,
                               n_samples=n_val_samples)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,
                                           collate_fn=my_collate_struct_to_seq)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False,
#                                           collate_fn=my_collate_struct_to_seq)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size, shuffle=False,
                                         collate_fn=my_collate_struct_to_seq)


def train_epoch(model, train_loader):
    model.train()
    losses = []
    h_losses = []
    accuracies = []

    for batch_idx, (dot_brackets, sequences, sequences_lengths) in enumerate(train_loader):
        dot_brackets = dot_brackets.to(opt.device)
        sequences = sequences.to(opt.device)
        sequences_lengths = sequences_lengths.to(opt.device)

        # Skip last batch if it does not have full size
        if dot_brackets.shape[0] < opt.batch_size:
            continue
        model.zero_grad()

        pred_sequences_scores = model(dot_brackets, sequences_lengths)

        # Loss is computed with respect to the target sequence
        loss = loss_function(pred_sequences_scores.view(-1, pred_sequences_scores.shape[2]),
                             sequences.view(-1))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # Metrics are computed with respect to generated folding
        if opt.evaluate_training:
            avg_h_loss, avg_accuracy = compute_metrics(target_dot_brackets=dot_brackets,
                                                       input_sequences=sequences,
                                                       pred_sequences_scores=pred_sequences_scores,
                                                       sequences_lengths=sequences_lengths,
                                                       verbose=opt.verbose)
            h_losses.append(avg_h_loss)
            accuracies.append(avg_accuracy)

    avg_loss = np.mean(losses)
    avg_h_loss = np.mean(h_losses)
    avg_accuracy = np.mean(accuracies)

    print("training loss is {}".format(avg_loss))
    print("training hamming loss: {}".format(avg_h_loss))
    print("accuracy: {}".format(avg_accuracy))

    return avg_loss.item(), avg_h_loss, avg_accuracy


def run(model, n_epochs, train_loader, results_dir, model_dir):
    print("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    loss, h_loss, accuracy = evaluate_struct_to_seq(model, train_loader, loss_function,
                                                    opt.batch_size, mode='train',
                                                    device=opt.device, verbose=opt.verbose)
    val_loss, val_h_loss, val_accuracy = evaluate_struct_to_seq(model, val_loader, loss_function,
                                                                opt.batch_size, mode='val',
                                                                device=opt.device, verbose=opt.verbose)

    train_losses = [loss]
    # test_losses = []
    val_losses = [val_loss]
    train_h_losses = [h_loss]
    # test_h_losses = []
    val_h_losses = [val_h_loss]
    train_accuracies = [accuracy]
    # test_accuracies = []
    val_accuracies = [val_accuracy]

    for epoch in range(n_epochs):
        start = time.time()
        print("Epoch {}: ".format(epoch + 1))

        loss, h_loss, accuracy = train_epoch(model, train_loader)
        # test_loss, test_h_loss, test_accuracy = evaluate_struct_to_seq(model, test_loader, loss_function,
        #                                                  batch_size, mode='test', device=opt.device)
        val_loss, val_h_loss, val_accuracy = evaluate_struct_to_seq(model, val_loader, loss_function,
                                                      opt.batch_size, mode='val',
                                                                    device=opt.device, verbose=opt.verbose)
        end = time.time()
        print("Epoch took {0:.2f} seconds".format(end - start))

        if not val_h_losses or val_h_loss > max(val_h_losses):
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

        if len(val_h_losses) > opt.early_stopping and min(val_h_losses[-opt.early_stopping:]) > \
                min(val_h_losses):
            print("Training terminated because of early stopping")
            print("Best val_loss: {}".format(min(val_losses)))
            print("Best val_h_loss: {}".format(min(val_h_losses)))
            print("Best val_accuracy: {}".format(max(val_accuracies)))
            break


def main():
    results_dir = '../results/{}/'.format(opt.model_name)
    model_dir = '../models/{}/'.format(opt.model_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(results_dir + 'hyperparams.txt', 'w') as f:
        f.write(str(opt))

    with open(results_dir + 'hyperparams.pkl', 'wb') as f:
        pickle.dump(opt, f)

    run(model, opt.n_epochs, train_loader, results_dir, model_dir)


if __name__ == "__main__":
    main()

