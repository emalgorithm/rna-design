import torch

from src.data_util.data_processing import seq_to_one_hot
from src.data_util.data_constants import word_to_ix


class GAN:
    def __init__(self, generator, discriminator, lr, device):
        self.generator = generator
        self.discriminator = discriminator

        # Optimizers
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()

        self.device = device

    def train(self, data):
        n_graphs = torch.unique(data.batch).shape[0]

        # Adversarial ground truths
        valid = torch.ones([n_graphs, 1]).to(self.device)
        fake = torch.zeros([n_graphs, 1]).to(self.device)

        # -----------------
        #  Train Generator
        # -----------------

        self.optimizer_G.zero_grad()

        # Generate a batch of images
        gen_sequences = self.generator(data)

        discriminator_fake_data = data.clone()
        discriminator_fake_data.x = gen_sequences
        # Loss measures generator's ability to fool the discriminator
        g_loss = self.adversarial_loss(self.discriminator(discriminator_fake_data), valid)

        g_loss.backward(retain_graph=True)
        self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()

        discriminator_real_data = data.clone()
        discriminator_real_data.x = seq_to_one_hot(discriminator_real_data.sequence,
                                                   len(word_to_ix))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(self.discriminator(discriminator_real_data), valid)
        fake_loss = self.adversarial_loss(self.discriminator(discriminator_fake_data), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        self.optimizer_D.step()

        return g_loss, d_loss
