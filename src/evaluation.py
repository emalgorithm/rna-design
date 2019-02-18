from sklearn.metrics import hamming_loss


def masked_hamming_loss(target, pred, ignore_idx=0):
    mask = target != ignore_idx
    return hamming_loss(target[mask], pred[mask])

