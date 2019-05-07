import torch

from src.data_util.data_processing import seq_to_one_hot
from src.data_util.data_constants import word_to_ix


class WGAN:
    def __init__(self, generator, discriminator, lr=0.00005, clip_value=0.01, n_critic=5, device="cpu"):
        self.generator = generator
        self.discriminator = discriminator

        # Optimizers
        self.optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
        self.optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

        self.clip_value = clip_value
        self.device = device
        self.i = 0
        self.n_critic = n_critic

    def train(self, data):
        g_loss = torch.Tensor([-10000]).to(self.device)
        gen_sequences = self.generator(data)

        discriminator_fake_data = data.clone()
        discriminator_fake_data.x = gen_sequences

        discriminator_real_data = data.clone()
        discriminator_real_data.x = seq_to_one_hot(discriminator_real_data.sequence,
                                                   len(word_to_ix))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()

        # Adversarial loss
        d_loss = -torch.mean(self.discriminator(discriminator_real_data)) + torch.mean(
            self.discriminator(discriminator_fake_data))

        d_loss.backward(retain_graph=True)
        self.optimizer_D.step()

        # Clip weights of discriminator
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)

        # Train the generator every n_critic iterations
        if self.i % self.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            self.optimizer_G.zero_grad()

            gen_sequences = self.generator(data)
            discriminator_fake_data = data.clone()
            discriminator_fake_data.x = gen_sequences
            # Adversarial loss
            g_loss = -torch.mean(self.discriminator(discriminator_fake_data))

            g_loss.backward()
            self.optimizer_G.step()

        self.i += 1

        return g_loss, d_loss
