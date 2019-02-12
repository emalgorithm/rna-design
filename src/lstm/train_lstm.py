from lstm.lstm_model import LSTMModel
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from sklearn.metrics import hamming_loss


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = pickle.load(open('../../data/temp.pkl', 'rb'))

word_to_ix = {"A": 0, "G": 1, "C": 2, "U": 3, 'T': 4} # TODO: Remove 'T' because it is for DNA
tag_to_ix = {".": 0, "(": 1, ")": 2}
ix_to_tag = {0: ".", 1: "(", 2: ")"}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 128

model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

training_data = training_data[:10]

for epoch in range(100):
    avg_loss = 0
    h_loss = 0
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        avg_loss += loss

        pred = tag_scores.max(1)[1]
        h_loss += hamming_loss(targets, pred)

    print("Epoch {}: loss is {}".format(epoch + 1, avg_loss / len(training_data)))
    print("Average hamming loss: {}".format(h_loss / len(training_data)))

with torch.no_grad():
    loss = 0
    for sentence, tags in training_data:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        model.hidden = model.init_hidden()

        tag_scores = model(sentence_in)
        pred = tag_scores.max(1)[1]
        loss += hamming_loss(prepare_sequence(tags, tag_to_ix), pred)
        # pred = [ix_to_tag[p.item()] for p in pred]

        # print("Real: {}".format(tags))
        # print("Pred: {}".format(''.join(pred)))
        # print()
    print("Average sequence length: {}".format(np.mean([len(seq) for seq, _ in training_data])))
    print("Average hamming loss: {}".format(loss / len(training_data)))
