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
            pe[pos, i] = ""
            pe[pos, i + 1] = ""
    return torch.from_numpy(pe).float()


def mask(x: Tensor, mask_value: float = 0.0, mask_diagonal: bool = False) -> Tensor:
    seq_len = x.size(1)
    indices = ""
    x[:, indices[0], indices[1]] = mask_value
    return x


def scaled_dot_product_attention(pad_mask: Tensor, query: Tensor, key: Tensor, value: Tensor, masking: bool) -> Tensor:
    dot_prod = query.bmm(key.transpose(1, 2))
    if masking:
        dot_prod = mask(dot_prod, float("-inf"))
    scale = ""
    pad_mask = pad_mask.unsqueeze(1).repeat(1, pad_mask.size(1), 1)
    scaled_dot_product = ""
    attention = ""
    return attention


class AttentionHead(nn.Module):

    def __init__(self, embedding_dim: int, query_dim: int, value_dim: int, masking: bool):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(embedding_dim, query_dim)
        self.k = nn.Linear(embedding_dim, query_dim)  # key_dim = query_dim
        self.v = nn.Linear(embedding_dim, value_dim)
        self.masking = masking

    def forward(self, pad_mask: Tensor, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(pad_mask, self.q(query), self.k(key), self.v(value), self.masking)


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, embedding_dim: int, query_dim: int, value_dim: int, masking: bool = False):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(embedding_dim, query_dim, value_dim, masking) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * value_dim, embedding_dim)

    def forward(self, pad_mask: Tensor, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        concat_heads = ""
        return self.linear(concat_heads)


class FeedForward(nn.Module):

    def __init__(self, input_dim: int = 512, hidden_dim: int = 2048):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            "",
            "",
            "",
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

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        x = ""
        x = ""
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

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        seq_len = x.size(1)
        embedding_dim = x.size(2)
        x += ""
        for layer in self.layers:
            x = layer(x, pad_mask)
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
            MultiHeadAttention(num_heads, embedding_dim, query_dim, value_dim, masking=True),
            input_dim=embedding_dim,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            FeedForward(embedding_dim, hidden_dim),
            input_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor, context: Tensor, dec_pad_mask: Tensor, enc_pad_mask: Tensor) -> Tensor:
        x = ""
        x = ""
        x = ""
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

    def forward(self, x: Tensor, context: Tensor, dec_pad_mask: Tensor, enc_pad_mask: Tensor) -> Tensor:
        seq_len, embedding_dim = x.size(1), x.size(2)
        x += ""
        for layer in self.layers:
            x = layer(x, context, enc_pad_mask, dec_pad_mask)
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
        # source 임베딩 --> encoder 입력
        # target 임베딩, source 임베딩 --> decoder 입력
        # decoder 출력을 one-hot 변환 --> softmax
        source_pad_mask = source == 0
        target_pad_mask = target == 0
        source = ""
        target = ""
        return target


# model config
tfm_config = TransformerConfig()

# train config
batch_size = 64

# test run
pad = torch.zeros([batch_size, 2]).long()
src = torch.randint(0, tfm_config.vocab_size, [batch_size, tfm_config.seq_len - 2])
tgt = torch.randint(0, tfm_config.vocab_size, [batch_size, tfm_config.seq_len - 2])
out = Transformer(tfm_config)(torch.cat([src, pad], dim=1), torch.cat([tgt, pad], dim=1))
print("source:", src.shape)
print("target:", tgt.shape)
print("output:", out.shape)
