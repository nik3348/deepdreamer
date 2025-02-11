import torch.nn as nn
from src.mha import MultiHeadAttention
from src.ff import FeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        layer_norm_eps=1e-5,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            d_model,
            d_model,
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )

        self.ff = FeedForward(
            d_model, dim_feedforward=dim_feedforward, dropout=dropout)

        self.norm1 = nn.RMSNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.RMSNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _sa_block(self, x, attn_mask, is_causal):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, is_causal=is_causal)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.ff(x)
        return self.dropout2(x)

    def forward(self, src, src_mask=None, is_causal=False):
        '''
        Arguments:
            src: (batch_size, seq_len, d_model)
            src_mask: (batch_size, seq_len, seq_len)
            is_causal: bool
        '''
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, is_causal)
        x = x + self._ff_block(self.norm2(x))

        return x
