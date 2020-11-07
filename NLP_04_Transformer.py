import random

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as ftn

# Created by Alex Kim
# main reference https://towardsdatascience.com/transformers-from-scratch-in-pytorch-8777e346ca51
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# TODO: decoder masking
# TODO: decoder last linear layer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def positional_encoding(seq_len: int, embedding_dim: int) -> Tensor:
    pe = np.zeros([seq_len, embedding_dim])
    for pos in range(seq_len):
        for i in range(0, embedding_dim, 2):
            pe[pos, i] = np.sin(pos / (1e+4 ** ((2 * i) / embedding_dim)))
            pe[pos, i + 1] = np.cos(pos / (1e+4 ** ((2 * (i + 1)) / embedding_dim)))
    return torch.from_numpy(pe)


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    dot_prod = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    attention = ftn.softmax(dot_prod / scale, dim=-1).bmm(value)
    return attention


class AttentionHead(nn.Module):

    def __init__(self, embedding_dim: int, query_dim: int, value_dim: int):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(embedding_dim, query_dim)
        self.k = nn.Linear(embedding_dim, query_dim)  # key_dim = query_dim
        self.v = nn.Linear(embedding_dim, value_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, embedding_dim: int, query_dim: int, value_dim: int):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(embedding_dim, query_dim, value_dim) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * value_dim, embedding_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        concat_heads = torch.cat([head(query, key, value) for head in self.heads], dim=-1)
        return self.linear(concat_heads)


class FeedForward(nn.Module):

    def __init__(self, input_dim: int = 512, hidden_dim: int = 2048):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.ff(x)


class Residual(nn.Module):

    def __init__(self, sublayer: nn.Module, input_dim: int = 512, dropout: float = 0.1):
        super(Residual, self).__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *x: Tensor) -> Tensor:
        return self.norm(x[-1] + self.dropout(self.sublayer(*x)))


class TransformerEncoderLayer(nn.Module):

    def __init__(
            self,
            embedding_dim: int = 512,
            num_heads: int = 6,
            hidden_dim: int = 2048,
            dropout: float = 0.1,
    ):
        super(TransformerEncoderLayer, self).__init__()
        query_dim = value_dim = embedding_dim // num_heads
        self.self_attention = Residual(
            MultiHeadAttention(num_heads, embedding_dim, query_dim, value_dim),
            input_dim=embedding_dim,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            FeedForward(embedding_dim, hidden_dim),
            input_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.self_attention(x, x, x)
        x = self.feed_forward(x)
        return x


class TransformerEncoder(nn.Module):

    def __init__(
            self,
            num_layers: int = 6,
            embedding_dim: int = 512,
            num_heads: int = 8,
            hidden_dim: int = 2048,
            dropout: float = 0.1,
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        embedding_dim = x.size(2)
        x += positional_encoding(seq_len, embedding_dim)
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
            self,
            embedding_dim: int = 512,
            num_heads: int = 6,
            hidden_dim: int = 2048,
            dropout: float = 0.1,
    ):
        super(TransformerDecoderLayer, self).__init__()
        query_dim = value_dim = embedding_dim // num_heads
        self.masked_attention = Residual(
            MultiHeadAttention(num_heads, embedding_dim, query_dim, value_dim),
            input_dim=embedding_dim,
            dropout=dropout,
        )
        self.self_attention = Residual(
            MultiHeadAttention(num_heads, embedding_dim, query_dim, value_dim),
            input_dim=embedding_dim,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            FeedForward(embedding_dim, hidden_dim),
            input_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        x = self.masked_attention(x, x, x)
        x = self.self_attention(context, context, x)
        x = self.feed_forward(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(
            self,
            num_layers: int = 6,
            embedding_dim: int = 512,
            num_heads: int = 8,
            hidden_dim: int = 2048,
            dropout: float = 0.1,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        seq_len, model_dim = x.size(1), x.size(2)
        x += positional_encoding(seq_len, model_dim)
        for layer in self.layers:
            x = layer(x, context)
        return torch.softmax(self.linear(x), dim=-1)


class Transformer(nn.Module):

    def __init__(
            self,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            embedding_dim: int = 512,
            num_heads: int = 6,
            hidden_dim: int = 2048,
            dropout: float = 0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        return self.decoder(target, self.encoder(source))


src = torch.rand(64, 16, 512)
tgt = torch.rand(64, 16, 512)
out = Transformer()(src, tgt)
print(out.shape)
