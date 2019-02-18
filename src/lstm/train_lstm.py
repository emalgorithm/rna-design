from data_util.data_processing import prepare_sequence, my_collate
from lstm.lstm_model import LSTMModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import hamming_loss
from data_util.rna_dataset import RNADataset
from torchvision import transforms
from visualization_util import plot_loss

# 'U' and 'T' in this sequences refer both to the base 'U'. 'T' is just used for convenience
word_to_ix = {"<PAD>": 0, "A": 1, "G": 2, "C": 3, "U": 4, 'T': 4}
tag_to_ix = {"<PAD>": 0, ".": 1, "(": 2, ")": 3}
ix_to_tag = {0: "<PAD>", 1: ".", 2: "(", 3: ")"}

# Model Definition
EMBEDDING_DIM = 6
HIDDEN_DIM = 64
batch_size = 32

model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), batch_size=batch_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Data Loading
x_transform = transforms.Lambda(lambda sequences: prepare_sequence(sequences, word_to_ix))
y_transform = transforms.Lambda(lambda sequences: prepare_sequence(sequences, tag_to_ix))

# train_set = RNADataset('../../data/temp_train/', x_transform=x_transform, y_transform=y_transform)
# test_set = RNADataset('../../data/temp_test/', x_transform=x_transform, y_transform=y_transform)
train_set = RNADataset('../../data/less_than_40/train/')
test_set = RNADataset('../../data/less_than_40/test/')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                           collate_fn=my_collate)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                          collate_fn=my_collate)


def train_epoch(model, train_loader):
    avg_loss = 0
    h_loss = 0
    for batch_idx, (sequences, dot_brackets) in enumerate(train_loader):
        # Skip last batch if it does not have full size
        if sequences.shape[0] < batch_size:
            continue
        print("Batch {}".format(batch_idx))
        model.zero_grad()

        base_scores = model(sequences)

        loss = loss_function(base_scores, dot_brackets.view(-1))
        loss.backward()
        optimizer.step()

        avg_loss += loss
        pred = base_scores.max(1)[1]
        h_loss += hamming_loss(dot_brackets.view(-1), pred)

    avg_loss /= len(train_loader)
    h_loss /= len(train_loader)

    print("training loss is {}".format(avg_loss))
    print("training hamming loss: {}".format(h_loss))

    return avg_loss, h_loss


def run(model, n_epochs, train_loader, test_loader):
    train_losses = []
    test_losses = []

    for epoch in range(n_epochs):
        print("Epoch {}: ".format(epoch + 1))

        loss, h_loss = train_epoch(model, train_loader)
        test_loss, test_h_loss = evaluate(model, test_loader)

        train_losses.append(loss)
        test_losses.append(test_loss)

        plot_loss(train_losses, test_losses)


def evaluate(model, test_loader):
    with torch.no_grad():
        loss = 0
        h_loss = 0

        for batch_idx, (sequences, dot_brackets) in enumerate(test_loader):
            # Skip last batch if it does not have full size
            if sequences.shape[0] < batch_size:
                continue

            base_scores = model(sequences)
            loss += loss_function(base_scores, dot_brackets.view(-1))
            pred = base_scores.max(1)[1]
            h_loss += hamming_loss(dot_brackets.view(-1), pred)

        loss /= len(test_loader)
        h_loss /= len(test_loader)

        print("test loss: {}".format(loss))
        print("test hamming loss: {}".format(h_loss))

        return loss, h_loss


run(model, 50, train_loader, test_loader)