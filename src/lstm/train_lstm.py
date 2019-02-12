from data_util.data_processing import prepare_sequence
from lstm.lstm_model import LSTMModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import hamming_loss
from data_util.rna_dataset import RNADataset
from torchvision import transforms

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
optimizer = optim.SGD(model.parameters(), lr=0.3)

# Data Loading
x_transform = transforms.Lambda(lambda sequences: prepare_sequence(sequences, word_to_ix))
y_transform = transforms.Lambda(lambda sequences: prepare_sequence(sequences, tag_to_ix))

train_set = RNADataset('../../data/temp_train/', x_transform=x_transform,
                       y_transform=y_transform)
test_set = RNADataset('../../data/temp_test/', x_transform=x_transform, y_transform=y_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

# Training
for epoch in range(100):
    avg_loss = 0
    h_loss = 0
    for batch_idx, (sequences, dot_brackets) in enumerate(train_loader):
        # Skip last batch if it does not have full size
        if sequences.shape[0] < batch_size:
            continue

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Run our forward pass.
        base_scores = model(sequences)

        # Step 3. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(base_scores, dot_brackets.view(-1))
        loss.backward()
        optimizer.step()
        avg_loss += loss

        pred = base_scores.max(1)[1]
        h_loss += hamming_loss(dot_brackets.view(-1), pred)

    print("Epoch {}: loss is {}".format(epoch + 1, avg_loss / len(train_loader)))
    print("Average hamming loss: {}".format(h_loss / len(train_loader)))

with torch.no_grad():
    h_loss = 0
    for batch_idx, (sequences, dot_brackets) in enumerate(test_loader):
        # Skip last batch if it does not have full size
        if sequences.shape[0] < batch_size:
            continue

        base_scores = model(sequences)
        pred = base_scores.max(1)[1]
        h_loss += hamming_loss(dot_brackets.view(-1), pred)

        pred = [ix_to_tag[p.item()] for p in pred]
        dot_brackets = [ix_to_tag[x.item()] for x in dot_brackets.view(-1)]
        print("Real: {}".format(''.join(dot_brackets)))
        print("Pred: {}".format(''.join(pred)))
        print()

    print("Average Test hamming loss: {}".format(h_loss / len(train_loader)))
