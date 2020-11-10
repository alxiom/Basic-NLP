import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch import optim
from torch.nn import functional as ftn
from torch.utils.data import Dataset, DataLoader

# Created by Alex Kim
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

train_gpt = True


@dataclass
class GPTConfig:
    seq_len: int = 16
    vocab_size: int = 8000
    embedding_drop_prob: float = 0.1
    residual_drop_prob: float = 0.1
    num_decoder_layers: int = 6
    num_heads: int = 8
    embedding_dim: int = 512
    hidden_dim: int = 2048


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4
    grad_norm_clip: float = 1.0
    checkpoint_path: str = None
    num_workers: int = 0  # for DataLoader


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
        skip = 0 if len(x) == 1 else 1
        return self.norm(x[skip] + self.dropout(self.sublayer(*x)))


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

    def forward(self, x: Tensor) -> Tensor:
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
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
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

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor):
        # embedding + position embedding
        # embedding dropout 적용
        # transformer decoder 입력
        # one-hot 벡터로 변환
        # cross entropy loss 계산
        x = self.embedding(x) + self.position_embedding[:, :x.size(1), :]
        x = self.embedding_dropout(x)
        x = self.decoder(x)
        x = self.linear(x)

        loss = None
        if y is not None:
            loss = ftn.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-1)
        return x, loss


# model config
model_config = GPTConfig()

# init model
gpt_model = GPT(model_config)

# train config
train_config = TrainConfig(batch_size=64)

# test run
src = torch.randint(0, model_config.vocab_size, [train_config.batch_size, model_config.seq_len])
tgt = torch.randint(0, model_config.vocab_size, [train_config.batch_size, model_config.seq_len])
out, ce_loss = gpt_model(src, tgt)
print("source:", src.shape)
print("target:", tgt.shape)
print("output:", out.shape)
print("loss:", ce_loss.item())


# addition demo
class AdditionDataset(Dataset):

    def __init__(self, n_digit, split):
        super(AdditionDataset, self).__init__()
        self.split = split  # train / valid
        self.n_digit = n_digit
        self.vocab_size = 10  # 10 digits [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # +1 due to potential carry overflow, but then -1 because very last digit doesn't plug back
        self.seq_len = n_digit + n_digit + n_digit + 1 - 1

        # split all addition problems to train / valid data
        problems = (10 ** self.n_digit) ** 2  # total number of possible combinations
        random_state = np.random.RandomState(42)  # make deterministic
        permute_problems = random_state.permutation(problems)
        num_valid = min(int(problems * 0.2), 1000)  # 20% of the whole dataset, or only up to 1000
        self.split_problems = permute_problems[:num_valid] if split == "valid" else permute_problems[num_valid:]

    def __getitem__(self, index):
        problem = self.split_problems[index]
        digit = 10 ** self.n_digit
        a = problem // digit
        b = problem % digit
        render = f"{a:0{self.n_digit}d}{b:0{self.n_digit}d}{a + b:0{self.n_digit + 1}d}"  # 03+25=28 --> "0325028"
        token_seq = [int(s) for s in render]  # ex) [0, 3, 2, 5, 0, 2, 8]
        x = torch.tensor(token_seq[:-1]).long()
        y = torch.tensor(token_seq[1:]).long()  # shift "right" to predict the next token
        y[:self.n_digit * 2 - 1] = -1  # we will only train in the output locations. -1 will mask loss to zero
        return x, y

    def __len__(self):
        return self.split_problems.size


sample_digit = 2
train_dataset = AdditionDataset(n_digit=sample_digit, split="train")
valid_dataset = AdditionDataset(n_digit=sample_digit, split="valid")

# train data sample
print(train_dataset[0])

# model config
model_config = GPTConfig(
    seq_len=train_dataset.seq_len,
    vocab_size=train_dataset.vocab_size,
    num_decoder_layers=2,
    num_heads=4,
    embedding_dim=128,
    hidden_dim=512,
)

# init model
gpt_model = GPT(model_config)

# train config
train_config = TrainConfig(
    epochs=80,
    batch_size=512,
    learning_rate=6e-4,
    num_workers=4,
    checkpoint_path="checkpoint",
)


class Trainer:

    def __init__(self, model, train_data, valid_data, config):
        super(Trainer, self).__init__()
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.n_digit = train_data.n_digit
        self.seq_len = train_data.seq_len
        self.config = config
        self.device = "cpu"
        self.global_step = 0
        self.start_epoch = 1
        self.epochs = config.epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def run(self):

        best_loss = float("inf")

        print("load valid set...")
        valid_data_loader = DataLoader(self.valid_data, batch_size=self.config.batch_size, shuffle=False)

        print("run 0 epoch...")
        self.model.eval()
        with torch.no_grad():
            valid_loss = self.run_epoch(valid_data_loader, "valid")
            print(f"Epoch: 0 / valid loss: {valid_loss:.4f}")

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_data_loader = DataLoader(
                self.train_data,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )

            print(f"run {epoch} epoch...")
            self.model.train()
            train_loss = self.run_epoch(train_data_loader, "train")
            print(f"Epoch: {epoch:2d} / train loss: {train_loss:.4f}")

            self.model.eval()
            with torch.no_grad():
                valid_loss = self.run_epoch(valid_data_loader, "valid")
                print(f"Epoch: {epoch:2d} / valid loss: {valid_loss:.4f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                self.save_checkpoint()

    def run_epoch(self, data_loader, mode):
        epoch_loss = 0.0
        epoch_count = 0

        for x, y in data_loader:
            batch_size = y.size(0)
            y_hat, batch_loss = self.model(x, y)

            if mode == "train":
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            epoch_loss = (epoch_loss * epoch_count + batch_loss.item() * batch_size) / (epoch_count + batch_size)
            epoch_count += batch_size
        return epoch_loss

    def save_checkpoint(self):
        if self.config.checkpoint_path is not None:
            print("save checkpoint...")
            torch.save(self.model.state_dict(), f"{self.config.checkpoint_path}/gpt.pt")


if train_gpt:
    Trainer(
        gpt_model,
        train_dataset,
        valid_dataset,
        train_config,
    ).run()

# load trained model
gpt_model.load_state_dict(torch.load("checkpoint/gpt.pt"))
gpt_model.eval()


def generate(model, x, step):
    seq_len = model.seq_len
    for _ in range(step):
        x_crop = x if x.size(1) <= seq_len else x[:, -seq_len:]  # crop left
        logit, _ = model(x_crop)
        logit = logit[:, -1, :]
        _, prediction = torch.topk(logit, k=1, dim=-1)
        x = torch.cat((x, prediction), dim=1)
    return x


def test_model(model, dataset, batch_size):
    results = []
    n_digit = dataset.n_digit
    for x, _ in DataLoader(dataset, batch_size=batch_size):
        query = x[:, :n_digit * 2]
        generate_answer = generate(model, query, n_digit + 1)
        answer = generate_answer[:, -(n_digit + 1):]
        base10 = torch.tensor([10 ** i for i in range(n_digit + 1)]).long().flip(0).unsqueeze(0)
        a = (query[:, :n_digit] * base10[:, (n_digit - 1):]).sum(1)
        b = (query[:, n_digit:n_digit * 2] * base10[:, (n_digit - 1):]).sum(1)
        gt = a + b
        answer = (answer * base10).sum(1)
        correct = answer == gt
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if not correct[i]:
                print(f"model prediction: {a[i]} + {b[i]} = {answer[i]} (GT = {gt[i]}) --> {correct[i]}")
    print(f"final score: {sum(results)} / {len(results)} = {np.mean(results) * 100:.2f} %% correct")


print("test on train set (inner test)")
test_model(gpt_model, train_dataset, batch_size=1024)
print("--------------------------------------------")
print("test on valid set")
test_model(gpt_model, valid_dataset, batch_size=1024)
