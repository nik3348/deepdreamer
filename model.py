import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.RMSNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Apply attention with causal mask
        attn_output, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            attn_mask=self.causal_mask(x.size(1), x.device)
        )
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x

    def causal_mask(self, seq_len, device):
        return torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(device)


class SimpleTransformer(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=4, num_layers=4):
        super().__init__()
        # Transformer blocks with multi-head attention
        self.layers = nn.ModuleList([
            TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(embedding_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


class RSSM(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=4, num_layers=4):
        super().__init__()
        self.proj_to_rssm = nn.Linear(embedding_dim, embedding_dim // 2)
        self.proj_from_rssm = nn.Linear(embedding_dim // 2, embedding_dim)
        self.rssm = SimpleTransformer(
            embedding_dim=embedding_dim // 2,
            num_heads=num_heads,
            num_layers=num_layers
        )

    def forward(self, z):
        z_down = self.proj_to_rssm(z)
        rssm_out = self.rssm(z_down)
        z = self.proj_from_rssm(rssm_out)
        return z


class SwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim * 2)
        self.linear2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x_proj = self.linear1(x)
        x_a, x_b = x_proj.chunk(2, dim=-1)
        x = x_a * torch.sigmoid(x_b)  # swish(x_b)
        return self.linear2(x)


class Model(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_attention_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1024, embedding_dim))
        self.dropout = nn.Dropout(dropout)

        self.encoder = SimpleTransformer(
            embedding_dim=embedding_dim,
            num_heads=num_attention_heads,
            num_layers=num_layers
        )
        self.decoder = SwiGLU(embedding_dim, vocab_size)

        self.rssm = RSSM(
            embedding_dim=embedding_dim,
            num_heads=num_attention_heads,
            num_layers=num_layers
        )

    def forward(self, input_ids):
        z = self.encode(input_ids)
        z_next_pred = self.rssm(z)
        x_pred = self.decoder(z)

        return z, z_next_pred, x_pred

    def encode(self, input_ids):
        x = self.embed(input_ids)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)

        z = self.encoder(x)
        return z

    def rollout_latent_future(self, z):
        z_next = self.rssm(z)
        x_pred = self.decoder(z_next)
        return z_next, x_pred
