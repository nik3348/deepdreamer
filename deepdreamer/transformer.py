import torch.nn as nn

from src.pe import PositionalEncoding
from src.encoder import EncoderLayer


class Transformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.embedding_layer = nn.Embedding(d_model, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.encoder_layer1 = EncoderLayer(d_model=d_model, nhead=8)
        self.encoder_layer2 = EncoderLayer(d_model=d_model, nhead=8)
        self.encoder_layer3 = EncoderLayer(d_model=d_model, nhead=8)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.positional_encoding(x.transpose(0, 1))
        x = x.transpose(0, 1)

        x = self.encoder_layer1(x, is_causal=True)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        return x
