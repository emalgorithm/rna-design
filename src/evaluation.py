from sklearn.metrics import hamming_loss
import numpy as np


def masked_hamming_loss(target, pred, ignore_idx=0):
    mask = target != ignore_idx
    return hamming_loss(target[mask], pred[mask])


def compute_accuracy(target, pred, ignore_idx=0):
    accuracy = 0
    for i in range(len(target)):
        mask = target[i] != ignore_idx
        accuracy += 1 if np.array_equal(target[i][mask], pred[i][mask]) else 0

    return accuracy / len(pred)

