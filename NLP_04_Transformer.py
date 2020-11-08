import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as ftn

# Created by Alex Kim
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TransformerConfig:
    batch_size: int = 64
    seq_len: int = 16
    vocab_size: int = 8000
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    embedding_dim: int = 512
    num_heads: int = 6
    hidden_dim: int = 2048
    dropout: float = 0.1


def positional_encoding(seq_len: int, embedding_dim: int) -> Tensor:
    pe = np.zeros([seq_len, embedding_dim])
    for pos in range(seq_len):
        for i in range(0, embedding_dim, 2):
            pe[pos, i] = np.sin(pos / (1e+4 ** ((2 * i) / embedding_dim)))
            pe[pos, i + 1] = np.cos(pos / (1e+4 ** ((2 * (i + 1)) / embedding_dim)))
    return torch.from_numpy(pe)


def mask(x: Tensor, mask_value: float = 0.0, mask_diagonal: bool = False) -> Tensor:
    seq_len = x.size(1)
    indices = torch.triu_indices(seq_len, seq_len, offset=0 if mask_diagonal else 1)
    x[:, indices[0], indices[1]] = mask_value
    return x


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, masking: bool) -> Tensor:
    dot_prod = query.bmm(key.transpose(1, 2))
    if masking:
        dot_prod = mask(dot_prod, float("-inf"))
    scale = query.size(-1) ** 0.5
    attention = ftn.softmax(dot_prod / scale, dim=-1).bmm(value)
    return attention


class AttentionHead(nn.Module):

    def __init__(self, embedding_dim: int, query_dim: int, value_dim: int, masking: bool):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(embedding_dim, query_dim)
        self.k = nn.Linear(embedding_dim, query_dim)  # key_dim = query_dim
        self.v = nn.Linear(embedding_dim, value_dim)
        self.masking = masking

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value), self.masking)


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, embedding_dim: int, query_dim: int, value_dim: int, masking: bool = False):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(embedding_dim, query_dim, value_dim, masking) for _ in range(num_heads)]
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

    def forward(self, x) -> Tensor:
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
            MultiHeadAttention(num_heads, embedding_dim, query_dim, value_dim, masking=True),
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

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        seq_len, embedding_dim = x.size(1), x.size(2)
        x += positional_encoding(seq_len, embedding_dim)
        for layer in self.layers:
            x = layer(x, context)
        return x


class Transformer(nn.Module):

    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.encoder = TransformerEncoder(
            num_layers=config.num_encoder_layers,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=config.num_decoder_layers,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        source = self.embedding(source)
        source = self.encoder(source)
        target = self.embedding(target)
        target = self.decoder(target, source)
        target = torch.matmul(target, self.embedding.weight.transpose(0, 1))
        target = torch.softmax(target, dim=-1)
        return target


tfm_config = TransformerConfig()
src = torch.randint(0, tfm_config.vocab_size, [tfm_config.batch_size, tfm_config.seq_len])
tgt = torch.randint(0, tfm_config.vocab_size, [tfm_config.batch_size, tfm_config.seq_len])
out = Transformer(tfm_config)(src, tgt)
print("source:", src.shape)
print("target:", tgt.shape)
print("output:", out.shape)
