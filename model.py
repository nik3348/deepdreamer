import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer import Transformer


class Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(2, d_model)
        self.gelu = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class World(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Actor(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Critic(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Model(nn.Module):
    def __init__(self, vocab_size, d_model, batch_size, seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.encoder = Encoder(d_model)
        self.decoder = Decoder(d_model)
        self.rssm = Transformer(d_model)
        self.world = World(d_model)
        self.actor = Actor(d_model, vocab_size)
        self.critic = Critic(d_model)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, obs, h, is_training=True):
        z = self.encoder(obs)
        probs = torch.softmax(z, dim=-1)
        z_sample = torch.argmax(probs, dim=-1)
        h_next = torch.cat((h, z_sample.unsqueeze(-1).unsqueeze(-1)), dim=1)
        world_loss = None

        if is_training:
            x = self.rssm(h)
            h_logits = self.world(x)
            world_loss = F.cross_entropy(
                h_logits.view(-1, h_logits.shape[-1]), h_next.view(-1)[1:])

        x_next = self.rssm(h_next)
        return h_next, self.actor(x_next), self.critic(x_next), z, world_loss

    def sample(self, logits):
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.vocab_size, dim=-1)

        ix = torch.multinomial(topk_probs, 1)
        xcol = topk_indices.gather(1, ix)

        return xcol

    def ae_loss(self, z, obss):
        obs_preds = self.decoder(z)
        return F.mse_loss(obs_preds, obss)

    def calc_loss(self, actions, rewards, dones, values):
        gamma_ = 0.99
        lambda_ = 0.95
        eta = 0.01

        actions = actions.reshape(
            self.batch_size, self.seq_len, self.vocab_size)
        rewards = rewards.reshape(self.batch_size, self.seq_len)
        dones = dones.reshape(self.batch_size, self.seq_len)
        values = values.reshape(self.batch_size, self.seq_len)

        lambda_returns = torch.zeros_like(rewards)
        lambda_returns[-1] = values[-1]

        for t in reversed(range(values.shape[0] - 1)):
            bootstrap = (1 - lambda_) * \
                values[t + 1] + lambda_ * lambda_returns[t + 1]
            lambda_returns[t] = rewards[t] + gamma_ * dones[t] * bootstrap

        lambda_returns = lambda_returns.detach()

        scaling_factor = torch.quantile(lambda_returns, 0.95) - torch.quantile(
            lambda_returns, 0.05
        )
        scaling_factor = torch.clamp(scaling_factor, min=1.0)
        scaled_returns = lambda_returns / scaling_factor
        policy_gradient_loss = -torch.sum(scaled_returns, dim=0).mean()

        # Shape [T, B, A]
        policy_probs = F.softmax(actions, dim=-1)
        uniform_probs = torch.full_like(policy_probs, 1.0 / 3)
        policy_probs = (1 - 0.01) * policy_probs + 0.01 * uniform_probs

        # Shape [T, B, A]
        # Shape [T, B]
        # Shape [B]
        policy_log_probs = F.log_softmax(actions, dim=-1)
        entropy = -(policy_probs * policy_log_probs).sum(dim=-1)
        entropy_regularization = -eta * torch.sum(entropy, dim=0).mean()

        actor_loss = policy_gradient_loss + entropy_regularization
        critic_loss = F.mse_loss(values, lambda_returns)
        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss, critic_loss
