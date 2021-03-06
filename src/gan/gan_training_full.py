import argparse
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch

from src.gan.gcn_generator import GCNGenerator
from src.gan.rnn_generator import RNNGenerator
from src.gan.gcn_discriminator import GCNDiscriminator
from src.gan.rnn_discriminator import RNNDiscriminator
from src.gan.fc_discriminator import FCDiscriminator
from src.data_util.rna_dataset_single_file import RNADatasetSingleFile
from src.data_util.data_processing import one_hot_embed_sequence, prepare_sequences, decode_sequence
from data_util.data_constants import word_to_ix, tag_to_ix, ix_to_word
import networkx as nx
from src.gcn.gcn_util import sparse_mx_to_torch_sparse_tensor
from src.util import dotbracket_to_graph


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
parser.add_argument('--seq_max_len', type=int, default=100, help='Maximum length of sequences '
                                                                 'used for training and testing')
parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to train on')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# The graph comes with no features, but the GCN needs features on each node.
# To overcome this, we choose a number of features, then create a noisy vector of that size and
# initialize all nodes with these features.
n_features = 128
r = np.random.normal(0, 1, n_features)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
# generator = GCNGenerator(n_features=n_features)
generator = RNNGenerator()
# Discriminator has 6 features which corresponds to the 1-hot encoding of the possible labels
# discriminator = FCDiscriminator(40)
discriminator = RNNDiscriminator()
# discriminator = Discriminator(n_features=len(word_to_ix))

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
# optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

n_train_samples = None if not opt.n_samples else int(opt.n_samples * 0.8)
n_val_samples = None if not opt.n_samples else int(opt.n_samples * 0.1)
train_set = RNADatasetSingleFile('../../data/sequences_with_folding_train.pkl', seq_max_len=40,
                                 seq_min_len=40, graph=False, n_samples=n_train_samples)
# test_set = RNADatasetSingleFile('../../data/sequences_with_folding_test.pkl',
#                                 seq_max_len=opt.seq_max_len, graph=False, n_samples=n_val_samples)
# val_set = RNADatasetSingleFile('../../data/sequences_with_folding_val.pkl',
#                                seq_max_len=opt.seq_max_len, graph=False, n_samples=n_val_samples)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)
# TODO: At the moment the graph does not distinguish between edge types

# ----------
#  Training
# ----------


for epoch in range(opt.n_epochs):
    generator.train()
    discriminator.train()
    for i, (seq, dot_bracket) in enumerate(train_loader):
        # For graph model
        # Batch contains a single element, extract it
        dot_bracket = dot_bracket[0]
        seq = seq[0]

        # For RNN Model
        targets, _ = prepare_sequences([seq], word_to_ix)
        sequences, sequences_lengths = prepare_sequences([dot_bracket], tag_to_ix)

        g = dotbracket_to_graph(dot_bracket)
        sample_y = nx.adjacency_matrix(g, nodelist=sorted(list(g.nodes())))
        adj = sparse_mx_to_torch_sparse_tensor(sample_y)

        x = one_hot_embed_sequence(seq, word_to_ix)
        hot_embedded_dot_bracket = one_hot_embed_sequence(dot_bracket, tag_to_ix)

        # Adversarial ground truths
        valid = Variable(Tensor(1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(1).fill_(0.0), requires_grad=False)

        # ---------------------
        # #  Train Discriminator
        # # ---------------------
        # # TODO: Discriminator is not learning. Even if we keep the generator fixed,
        # #  the discriminator does not improve its loss
        #
        # optimizer_D.zero_grad()
        #
        # # Sample noise as generator input
        # # TODO: Reinsert noise here
        # z = Variable(Tensor(r))
        # # z = Variable(Tensor(np.random.normal(0, 1, n_features)))
        #
        # # Generate graph features
        # generated_x = generator(sequences, sequences_lengths, z)
        # # generated_x = generator(adj, z, n_nodes=len(x))
        # print("Real:")
        # print(x)
        # print(seq)
        # discriminator_real_score = discriminator(x, adj)
        # print("Discriminator real score: {}".format(discriminator_real_score))
        # print("Generated:")
        # # TODO: generated_x is continuous softmax, whereas x is one-hot. So for the discriminator
        # #  it is very easy to distinguish between them, as the generator can just learn to accept
        # #  only one-hot encodings. If I transform the generated matrix into one-hot using a max,
        # #  I'm not sure how the learning would go (gradient)
        # print(generated_x)
        # pred = generated_x.max(1)[1].numpy()
        # pred = decode_sequence(pred, ix_to_word)
        # print(pred)
        # discriminator_generated_score = discriminator(generated_x, adj)
        # print("Discriminator generated score: {}".format(discriminator_generated_score))
        # # Adversarial loss
        # d_loss = torch.mean(discriminator_generated_score)
        # # d_loss = -torch.mean(discriminator_real_score) + torch.mean(discriminator_generated_score)
        #
        # d_loss.backward()
        # optimizer_D.step()
        #
        # # Clip weights of discriminator
        # # for p in discriminator.parameters():
        # #     p.data.clamp_(-opt.clip_value, opt.clip_value)
        #
        # # Train the generator every n_critic iterations
        # if i % opt.n_critic == 0:
        #     # -----------------
        #     #  Train Generator
        #     # -----------------
        #
        #     optimizer_G.zero_grad()
        #
        #     # Generate graph features
        #     generated_x = generator(sequences, sequences_lengths, z)
        #     # generated_x = generator(adj, z, n_nodes=len(x))
        #     # Adversarial loss
        #     g_loss = -torch.mean(discriminator(generated_x, adj))
        #
        #     # g_loss.backward()
        #     # optimizer_G.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        # # TODO: Reinsert noise here
        # z = Variable(Tensor(r))
        z = Variable(Tensor(np.random.normal(0, 1, n_features)))

        # Generate graph features
        generated_x = generator(hot_embedded_dot_bracket, sequences_lengths, z)
        # generated_x = generator(adj, z, n_nodes=len(x))
        print("Real:")
        print(x)
        print(seq)
        discriminator_real_score = discriminator(x, adj)
        print("Discriminator real score: {}".format(discriminator_real_score))
        print("Generated:")
        # TODO: generated_x is continuous softmax, whereas x is one-hot. So for the discriminator
        #  it is very easy to distinguish between them, as the generator can just learn to accept
        #  only one-hot encodings. If I transform the generated matrix into one-hot using a max,
        #  I'm not sure how the learning would go (gradient)
        print(generated_x)
        pred = generated_x.max(1)[1].numpy()
        pred = decode_sequence(pred, ix_to_word)
        print(pred)
        discriminator_generated_score = discriminator(generated_x, adj)
        print("Discriminator generated score: {}".format(discriminator_generated_score))

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(generated_x, adj), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(x, adj), valid)
        fake_loss = adversarial_loss(discriminator(generated_x.detach(), adj), fake)
        # d_loss = real_loss / 2
        d_loss = (real_loss + fake_loss) / 2

        if epoch < 1:
            d_loss.backward()
            optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(train_loader),
                                                            d_loss.item(), g_loss.item()))

        batches_done = epoch * len(train_loader) + i

        # if batches_done % opt.sample_interval == 0:
        #     # Evaluate

# for i, (seq, adj) in enumerate(train_loader):
#     adj = adj[0]
#     g = dotbracket_to_graph(adj)
#     sample_y = nx.adjacency_matrix(g, nodelist=sorted(list(g.nodes())))
#     adj = sparse_mx_to_torch_sparse_tensor(sample_y)
#
#     x = one_hot_embed_sequence(seq[0], word_to_ix)
#     # Adversarial ground truths
#     valid = Variable(Tensor(1).fill_(1.0), requires_grad=False)
#     fake = Variable(Tensor(1).fill_(0.0), requires_grad=False)
#
#     # ---------------------
#     #  Train Discriminator
#     # ---------------------
#
#     optimizer_D.zero_grad()
#
#     # Sample noise as generator input
#     z = Variable(Tensor(np.random.normal(0, 1, n_features)))
#
#     # Generate a batch of images
#     generated_x = generator(adj, z, n_nodes=len(x))
