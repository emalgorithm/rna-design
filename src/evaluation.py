from sklearn.metrics import hamming_loss
import numpy as np
import torch


def masked_hamming_loss(target, pred, ignore_idx=0):
    mask = target != ignore_idx
    return hamming_loss(target[mask], pred[mask])


def compute_accuracy(target, pred, ignore_idx=0):
    accuracy = 0
    for i in range(len(target)):
        mask = target[i] != ignore_idx
        accuracy += 1 if np.array_equal(target[i][mask], pred[i][mask]) else 0

    return accuracy / len(pred)


def evaluate(model, test_loader, loss_function, batch_size, mode='test', device='cpu'):
    model.eval()
    with torch.no_grad():
        loss = 0
        h_loss = 0
        accuracy = 0

        for batch_idx, (sequences, dot_brackets, sequences_lengths) in enumerate(test_loader):
            sequences = sequences.to(device)
            dot_brackets = dot_brackets.to(device)
            sequences_lengths = sequences_lengths.to(device)

            # Skip last batch if it does not have full size
            if sequences.shape[0] < batch_size:
                continue

            base_scores = model(sequences, sequences_lengths)

            loss += loss_function(base_scores, dot_brackets.view(-1))
            pred = base_scores.max(1)[1]
            h_loss += masked_hamming_loss(dot_brackets.view(-1).cpu().numpy(), pred.cpu().numpy())
            accuracy += compute_accuracy(dot_brackets.cpu().numpy(), pred.view_as(
                dot_brackets).cpu().numpy())

        loss /= len(test_loader)
        h_loss /= len(test_loader)
        accuracy /= len(test_loader)

        print("{} loss: {}".format(mode, loss))
        print("{} hamming loss: {}".format(mode, h_loss))
        print("{} accuracy: {}".format(mode, accuracy))

        return loss.item(), h_loss, accuracy
