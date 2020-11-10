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

dump_text = True
train_tokenizer = True
train_bert = True


@dataclass
class BertConfig:
    seq_len: int = 16
    vocab_size: int = 8000
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    embedding_dim: int = 32
    num_heads: int = 6
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
        q, a, is_next_label = self.nsp_data(item)
        q_random, q_label = self.mlm_data(q)
        a_random, a_label = self.mlm_data(a)

        q = [special.index("<cls>")] + q_random + [special.index("<sep>")]
        a = a_random + [special.index("<eos>")]

        q_label = [special.index("<pad>")] + q_label + [special.index("<pad>")]
        a_label = a_label + [special.index("<pad>")]

        bert_input = (q + a)[:self.seq_len]
        bert_label = (q_label + a_label)[:self.seq_len]
        segment_label = ([1 for _ in range(len(q))] + [2 for _ in range(len(a))])[:self.seq_len]

        padding = [special.index("<pad>") for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {
            "bert_input": bert_input,
            "bert_label": bert_label,
            "segment_label": segment_label,
            "is_next": is_next_label,
        }
        return {key: torch.tensor(value) for key, value in output.items()}

    def mlm_data(self, text):
        tokens = self.tokenizer.encode(text).ids
        labels = []
        for i in range(len(tokens)):
            masking_prob = random.random()
            if masking_prob < 0.15:  # 전체의 15% masking
                labels.append(tokens[i])
                mask_type_prob = random.random()
                if mask_type_prob < 0.8:  # 마스킹 대상 중 80% 마스킹
                    tokens[i] = special.index("<mask>")
                elif mask_type_prob < 0.9:  # 마스킹 대상 중 10% random replace
                    tokens[i] = random.randrange(self.tokenizer.get_vocab_size())
            else:
                labels.append(0)
        return tokens, labels

    def nsp_data(self, item):
        q, a = self.get_corpus_line(item)

        # query, answer, label(isNext: 1, isNotNext: 0)
        if random.random() > 0.5:
            return q, a, 1
        else:
            return q, self.get_random_answer_line(), 0

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
        self.embedding_dim = config.embedding_dim
        self.vocab_size = config.vocab_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.segment_embedding = nn.Embedding(3, config.embedding_dim, padding_idx=0)
        self.encoder = TransformerEncoder(
            num_layers=config.num_encoder_layers,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

    def forward(self, x: Tensor, segment_label: Tensor):
        pad_mask = x == 0
        x_token_embedding = self.token_embedding(x)
        seq_len = x_token_embedding.size(1)
        embedding_dim = x_token_embedding.size(2)
        x_segment_embedding = self.segment_embedding(segment_label)
        x = x_token_embedding + x_segment_embedding + positional_encoding(seq_len, embedding_dim)
        x = self.encoder(x, pad_mask)
        return x


class MaskedLanguageModel(nn.Module):

    def __init__(self, embedding_dim: int, vocab_size: int):
        super(MaskedLanguageModel, self).__init__()
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return self.linear(x)


class NextSentencePrediction(nn.Module):

    def __init__(self, embedding_dim: int):
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(embedding_dim, 2)

    def forward(self, x):
        return self.linear(x[:, 0])


class BertLanguageModel(nn.Module):

    def __init__(self, bert_model: Bert):
        super(BertLanguageModel, self).__init__()
        self.bert_model = bert_model
        self.mlm = MaskedLanguageModel(self.bert_model.embedding_dim, self.bert_model.vocab_size)
        self.nsp = NextSentencePrediction(self.bert_model.embedding_dim)

    def forward(self, x, segment_label):
        x = self.bert_model(x, segment_label)
        return self.mlm(x), self.nsp(x)


train_dataset = BertDataset(chat_corpus, chat_tokenizer, BertConfig.seq_len)

bert = Bert(BertConfig)
bert_lm = BertLanguageModel(bert)

train_config = TrainConfig(
    epochs=10,
    batch_size=128,
    learning_rate=1e-4,
    num_workers=4,
    checkpoint_path="checkpoint",
)


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
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_data_loader = DataLoader(
                self.train_data,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )

            print(f"run {epoch} epoch...")
            self.model.train()
            mlm_loss, nsp_loss = self.run_epoch(train_data_loader)
            total_loss = mlm_loss + nsp_loss
            print(f"Epoch: {epoch:2d} / MLM: {mlm_loss:.4f} / NSP: {nsp_loss:.4f} / total loss: {total_loss:.4f}")

        self.save_checkpoint()

    def run_epoch(self, data_loader):
        epoch_mlm_loss = 0.0
        epoch_nsp_loss = 0.0
        epoch_count = 0

        for data in data_loader:
            batch_size = len(data)
            mlm_output, nsp_output = self.model(data["bert_input"], data["segment_label"])
            mlm_loss = self.criterion(mlm_output.transpose(1, 2), data["bert_label"])
            nsp_loss = self.criterion(nsp_output, data["is_next"])
            loss = mlm_loss + nsp_loss
            self.optimizer_schedule.zero_grad()
            loss.backward()
            self.optimizer_schedule.step_and_update_lr()

            epoch_mlm_loss = (epoch_mlm_loss * epoch_count + mlm_loss.item() * batch_size) / (epoch_count + batch_size)
            epoch_nsp_loss = (epoch_nsp_loss * epoch_count + nsp_loss.item() * batch_size) / (epoch_count + batch_size)
            epoch_count += batch_size
        return epoch_mlm_loss, epoch_nsp_loss

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
            np.power(self.warmup_steps, -1.5) * self.current_steps])

    def update_learning_rate(self):
        self.current_steps += 1
        lr = self.init_lr * self.get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


if train_bert:
    Trainer(
        bert,
        train_dataset,
        train_config,
    ).run()
