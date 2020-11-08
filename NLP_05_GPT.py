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


@dataclass
class GPTConfig:
    batch_size: int = 64
    seq_len: int = 16
    vocab_size: int = 8000
    embedding_drop_prob: float = 0.1
    residual_drop_prob: float = 0.1
    num_decoder_layers: int = 6
    num_heads: int = 8
    embedding_dim: int = 512
    hidden_dim: int = 2048


def mask(x: Tensor, mask_value: float = 0.0):
    seq_len = x.size(1)
    indices = torch.triu_indices(seq_len, seq_len, offset=1)
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


class TransformerDecoderLayer(nn.Module):

    def __init__(self, config):
        super(TransformerDecoderLayer, self).__init__()
        embedding_dim = config.embedding_dim
        num_heads = config.num_heads
        hidden_dim = config.hidden_dim
        residual_dropout = config.residual_drop_prob
        query_dim = value_dim = embedding_dim // num_heads
        self.masked_attention = Residual(
            MultiHeadAttention(num_heads, embedding_dim, query_dim, value_dim, masking=True),
            input_dim=embedding_dim,
            dropout=residual_dropout,
        )

        self.feed_forward = Residual(
            FeedForward(embedding_dim, hidden_dim),
            input_dim=embedding_dim,
            dropout=residual_dropout,
        )

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        x = self.masked_attention(x, x, x)
        x = self.feed_forward(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        num_layers = config.num_decoder_layers
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super(GPT, self).__init__()
        self.seq_len = config.seq_len
        self.embedding = nn.Linear(config.vocab_size, config.embedding_dim, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.seq_len, config.embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(config.embedding_drop_prob)
        self.decoder = TransformerDecoder(config)
        self.linear = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x) + self.position_embedding
        x = self.embedding_dropout(x)
        x = self.decoder(x)
        x = self.linear(x)
        return x


gpt_config = GPTConfig()
src = torch.rand(gpt_config.batch_size, gpt_config.seq_len, gpt_config.vocab_size)
out = GPT(gpt_config)(src)
print(out.shape)
