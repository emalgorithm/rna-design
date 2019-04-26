from sklearn.metrics import hamming_loss
import numpy as np
import torch
import RNA
from src.data_util.data_processing import one_hot_embed_sequence, prepare_sequences, decode_sequence
from src.data_util.data_constants import word_to_ix, tag_to_ix, ix_to_word, ix_to_tag


def masked_hamming_loss(target, pred, ignore_idx=0):
    mask = target != ignore_idx
    return hamming_loss(target[mask], pred[mask])


def compute_accuracy(target, pred, ignore_idx=0):
    accuracy = 0
    for i in range(len(target)):
        mask = target[i] != ignore_idx
        accuracy += 1 if np.array_equal(target[i][mask], pred[i][mask]) else 0

    return accuracy / len(pred)


def compute_metrics(target_dot_brackets, input_sequences, pred_sequences_scores, sequences_lengths,
                    verbose=False):
    dot_brackets_strings = [decode_sequence(dot_bracket.cpu().numpy()[:sequences_lengths[
        i]], ix_to_tag) for i, dot_bracket in enumerate(target_dot_brackets)]
    sequences_strings = [decode_sequence(sequence.cpu().numpy()[:sequences_lengths[
        i]], ix_to_word) for i, sequence in enumerate(input_sequences)]

    pred_sequences_np = pred_sequences_scores.max(2)[1].cpu().numpy()
    pred_sequences_strings = [decode_sequence(pred[:sequences_lengths[i]], ix_to_word) for i,
                                                          pred in enumerate(pred_sequences_np)]
    pred_dot_brackets_strings = [RNA.fold(pred_sequences_strings[i])[0] for
                                 i, pred_sequence in enumerate(pred_sequences_strings)]

    h_loss = np.mean([hamming_loss(list(dot_brackets_strings[i]),
                                   list(pred_dot_brackets_strings[i])) for i in range(len(
        pred_dot_brackets_strings))])
    accuracy = np.mean([1 if (dot_brackets_strings[i] == pred_dot_brackets_strings[i]) else 0 for i
                       in range(len(pred_dot_brackets_strings))])

    if verbose:
        for i in range(len(dot_brackets_strings)):
            print("REAL SEQUENCE: {}".format(sequences_strings[i]))
            print("PRED SEQUENCE: {}".format(pred_sequences_strings[i]))
            print("REAL: {}".format(dot_brackets_strings[i]))
            print("PRED: {}".format(pred_dot_brackets_strings[i]))
            print()

    return h_loss, accuracy


def evaluate(model, test_loader, loss_function, batch_size, mode='test', device='cpu'):
    model.eval()
    with torch.no_grad():
        loss = 0
        avg_h_loss = 0
        avg_accuracy = 0

        for batch_idx, (sequences, dot_brackets, sequences_lengths) in enumerate(test_loader):
            sequences = sequences.to(device)
            dot_brackets = dot_brackets.to(device)
            sequences_lengths = sequences_lengths.to(device)

            # Skip last batch if it does not have full size
            if sequences.shape[0] < batch_size:
                continue

            base_scores = model(sequences, sequences_lengths)

            loss += loss_function(base_scores.view(-1, base_scores.shape[2]), dot_brackets.view(-1))
            avg_h_loss, avg_accuracy = compute_metrics(base_scores, dot_brackets)
            avg_h_loss += avg_h_loss
            avg_accuracy += avg_accuracy

        loss /= len(test_loader)
        avg_h_loss /= len(test_loader)
        avg_accuracy /= len(test_loader)

        print("{} loss: {}".format(mode, loss))
        print("{} hamming loss: {}".format(mode, avg_h_loss))
        print("{} accuracy: {}".format(mode, avg_accuracy))

        return loss, avg_h_loss, avg_accuracy


def evaluate_struct_to_seq(model, test_loader, loss_function, batch_size, mode='test',
                           device='cpu'):
    model.eval()
    with torch.no_grad():
        loss = 0
        avg_h_loss = 0
        avg_accuracy = 0

        for batch_idx, (dot_brackets, sequences, sequences_lengths) in enumerate(test_loader):
            dot_brackets = dot_brackets.to(device)
            sequences = sequences.to(device)
            sequences_lengths = sequences_lengths.to(device)

            # Skip last batch if it does not have full size
            if dot_brackets.shape[0] < batch_size:
                continue

            base_scores = model(dot_brackets, sequences_lengths)

            loss += loss_function(base_scores.view(-1, base_scores.shape[2]), dot_brackets.view(-1))
            avg_h_loss, avg_accuracy = compute_metrics(target_dot_brackets=dot_brackets,
                                                       input_sequences=sequences,
                                                       pred_sequences_scores=base_scores,
                                                       sequences_lengths=sequences_lengths,
                                                       verbose=True)
            avg_h_loss += avg_h_loss
            avg_accuracy += avg_accuracy

        loss /= len(test_loader)
        avg_h_loss /= len(test_loader)
        avg_accuracy /= len(test_loader)

        print("{} loss: {}".format(mode, loss))
        print("{} hamming loss: {}".format(mode, avg_h_loss))
        print("{} accuracy: {}".format(mode, avg_accuracy))

        return loss, avg_h_loss, avg_accuracy
