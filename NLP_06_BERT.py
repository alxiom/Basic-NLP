import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn
from torch import optim
from torch.nn import functional as ftn
from torch.utils.data import Dataset, DataLoader
from tokenizers import CharBPETokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Created by Alex Kim
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

dump_text = False
train_tokenizer = False
train_bert = False


@dataclass
class BertConfig:
    seq_len: int = 16
    vocab_size: int = 8000
    num_encoder_layers: int = 6
    embedding_dim: int = 512
    num_heads: int = 8
    hidden_dim: int = 2048
    dropout: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    checkpoint_path: str = None
    num_workers: int = 0  # for DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
special = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<cls>", "<mask>"]

chat_corpus = pd.read_csv("data/chat.csv", header=0).sample(1000).to_numpy()
if dump_text:
    with open("data/chat_sample.txt", "w", encoding="utf-8") as f:
        for row_index in range(len(chat_corpus)):
            row = chat_corpus[row_index]
            chat_query = row[0]
            chat_answer = row[1]
            f.write(f"{chat_query} {chat_answer}\n")

# chat text --> vocab.json, merges.txt
if train_tokenizer:
    chat_tokenizer = CharBPETokenizer()
    chat_tokenizer.train(files=["data/chat_sample.txt"], vocab_size=8000, special_tokens=special, min_frequency=3)
    chat_tokenizer.save_model(f"data")

chat_tokenizer = CharBPETokenizer(vocab="data/vocab.json", merges="data/merges.txt")
tokenize_sample_text = chat_tokenizer.encode("인연이 있다고 생각해?")
print(tokenize_sample_text.tokens)
print("--")

recover = chat_tokenizer.decode(tokenize_sample_text.ids)
print(recover)
print("--")

print(chat_tokenizer.get_vocab_size())


class BertDataset(Dataset):

    def __init__(self, corpus, tokenizer, seq_len):
        super(BertDataset, self).__init__()
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return self.corpus_size

    def __getitem__(self, item):
        return {}

    def mlm_data(self, text):
        return "", ""

    def nsp_data(self, item):
        return "", "", 0

    def get_corpus_line(self, item):
        return self.corpus[item][0], self.corpus[item][1]

    def get_random_answer_line(self):
        return self.corpus[random.randrange(self.corpus_size)][1]


def positional_encoding(seq_len: int, embedding_dim: int) -> Tensor:
    pe = np.zeros([seq_len, embedding_dim])
    for pos in range(seq_len):
        for i in range(0, embedding_dim, 2):
            pe[pos, i] = np.sin(pos / (1e+4 ** ((2 * i) / embedding_dim)))
            pe[pos, i + 1] = np.cos(pos / (1e+4 ** ((2 * (i + 1)) / embedding_dim)))
    return torch.from_numpy(pe).float()


def mask(x: Tensor, mask_value: float = 0.0, mask_diagonal: bool = False) -> Tensor:
    seq_len = x.size(1)
    indices = torch.triu_indices(seq_len, seq_len, offset=0 if mask_diagonal else 1)
    x[:, indices[0], indices[1]] = mask_value
    return x


def scaled_dot_product_attention(pad_mask: Tensor, query: Tensor, key: Tensor, value: Tensor, masking: bool) -> Tensor:
    dot_prod = query.bmm(key.transpose(1, 2))
    if masking:
        dot_prod = mask(dot_prod, float("-inf"))
    scale = query.size(-1) ** 0.5
    pad_mask = pad_mask.unsqueeze(1).repeat(1, pad_mask.size(1), 1)
    scaled_dot_product = (dot_prod / scale).masked_fill_(pad_mask, -1e+9)
    attention = ftn.softmax(scaled_dot_product, dim=-1).bmm(value)
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
        concat_heads = torch.cat([head(pad_mask, query, key, value) for head in self.heads], dim=-1)
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
        skip = 0 if len(x) == 1 else 1
        return self.norm(x[skip] + self.dropout(self.sublayer(*x)))


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
        x = self.self_attention(pad_mask, x, x, x)
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

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        seq_len = x.size(1)
        embedding_dim = x.size(2)
        x += positional_encoding(seq_len, embedding_dim)
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x


class Bert(nn.Module):

    def __init__(self, config):
        super(Bert, self).__init__()


class MaskedLanguageModel(nn.Module):

    def __init__(self, embedding_dim: int, vocab_size: int):
        super(MaskedLanguageModel, self).__init__()


class NextSentencePrediction(nn.Module):

    def __init__(self, embedding_dim: int):
        super(NextSentencePrediction, self).__init__()


class BertLanguageModel(nn.Module):

    def __init__(self, bert_model: Bert):
        super(BertLanguageModel, self).__init__()


class Trainer:

    def __init__(self, bert_model, train_data, config):
        super(Trainer, self).__init__()
        self.bert_model = bert_model
        self.model = BertLanguageModel(self.bert_model)
        self.train_data = train_data
        self.config = config
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.optimizer_schedule = ScheduleOptimizer(
            self.optimizer,
            self.bert_model.embedding_dim,
            warmup_steps=config.warmup_steps,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=special.index("<pad>"))
        self.start_epoch = 1
        self.epochs = config.epochs

    def run(self):
        None

    def run_epoch(self, data_loader):
        return 0.0

    def save_checkpoint(self):
        if self.config.checkpoint_path is not None:
            print("save checkpoint...")
            torch.save(self.model.state_dict(), f"{self.config.checkpoint_path}/bert.pt")


class ScheduleOptimizer:

    def __init__(self, optimizer, embedding_dim, warmup_steps):
        super(ScheduleOptimizer, self).__init__()
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_steps = 0
        self.init_lr = np.power(embedding_dim, -0.5)

    def step_and_update_lr(self):
        self.update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr_scale(self):
        return np.min([
            np.power(self.current_steps, -0.5),
            np.power(self.warmup_steps, -1.5) * self.current_steps,
        ])

    def update_learning_rate(self):
        self.current_steps += 1
        lr = self.init_lr * self.get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


# prepare dataset
sequence_length = 16
train_dataset = BertDataset(chat_corpus, chat_tokenizer, sequence_length)

# train data sample
print(train_dataset[0])

# model config
model_config = BertConfig(
    seq_len=sequence_length,
    vocab_size=chat_tokenizer.get_vocab_size(),
    num_encoder_layers=4,
    embedding_dim=128,
    num_heads=4,
    hidden_dim=512,
)

# init model
bert = Bert(model_config)
bert_lm = BertLanguageModel(bert)

# train config
train_config = TrainConfig(
    epochs=10,
    batch_size=512,
    learning_rate=1e-4,
    num_workers=4,
    checkpoint_path="checkpoint",
)

if train_bert:
    Trainer(
        bert,
        train_dataset,
        train_config,
    ).run()
