import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as ftn
from torch.utils.data import Dataset, DataLoader
from tokenizers import CharBPETokenizer

# Created by Alex Kim
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
special = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<cls>", "<mask>"]

train_seq2seq_attention = True

tokenizer = CharBPETokenizer(vocab="data/vocab.json", merges="data/merges.txt")

train_data = pd.read_csv("data/chat_sample.csv", header=0)
print(train_data.head(5))
print(len(train_data))
print("--")

query_tokens = []
answer_tokens = []
for i in range(len(train_data)):
    row = train_data.loc[i]
    query = row["Q"]
    answer = row["A"]

    tokenize_query = tokenizer.encode(query)
    tokenize_answer = tokenizer.encode(answer)

    query_tokens.append(tokenize_query.ids)
    answer_tokens.append(tokenize_answer.ids)


class LoadDataset(Dataset):

    def __init__(self, x_data, y_data):
        super(LoadDataset, self).__init__()
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return len(self.y_data)


class MaxPadBatch:

    def __init__(self, max_len=24):
        super(MaxPadBatch, self).__init__()
        self.max_len = max_len

    def __call__(self, batch):
        batch_x = []
        batch_y = []
        for x, y in batch:
            batch_x.append(torch.tensor(x).long())
            batch_y.append(torch.tensor([special.index("<bos>")] + y + [special.index("<eos>")]).long())
        pad_index = special.index("<pad>")
        pad_x = [ftn.pad(item, [0, self.max_len - item.shape[0]], value=pad_index).detach() for item in batch_x]
        pad_y = [ftn.pad(item, [0, self.max_len - item.shape[0]], value=pad_index).detach() for item in batch_y]
        return torch.stack(pad_x), torch.stack(pad_y), len(batch)


max_seq_length = 20
chat_dataset = LoadDataset(query_tokens, answer_tokens)
chat_data_loader = DataLoader(chat_dataset, batch_size=32, collate_fn=MaxPadBatch(max_seq_length))


class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)

    def forward(self, x, embedding):
        # x: [batch, seq_length]
        x = embedding(x)
        x, hidden = self.rnn(x)
        return x, hidden


class Decoder(nn.Module):

    def __init__(self, output_size, embedding_size, hidden_size):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)

    def forward(self, x, hidden, embedding):
        # x: [batch] --> need second dimension as 1
        # hidden: [encoder_layers = 1, batch, hidden_dim]
        x = x.unsqueeze(1)
        x = embedding(x)
        x, hidden = self.rnn(x, hidden)
        return x, hidden


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_output, decoder_output):
        # 이번 decoder 출력이 encoder 모든 출력들과 얼마나 강한 관계가 있는지 측정
        # 이번 decoder 출력과 encoder 모든 출력과 dot product 실행 --> sequence of scala (=attention score)
        # attention score --> softmax --> attention weight
        # 위에서 구한 강도에 따라서 encoder 모든 출력을 weight sum --> context_vector
        attention_score = torch.bmm(decoder_output, encoder_output.transpose(1, 2))
        attention_weight = self.softmax(attention_score)
        context_vector = torch.bmm(attention_weight, encoder_output)
        return context_vector


class Seq2SeqAttention(nn.Module):

    def __init__(self, encoder, decoder, attention):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.embedding = nn.Embedding(self.encoder.input_size, self.encoder.embedding_size)
        self.target_vocab_size = self.decoder.output_size
        self.linear = nn.Linear(self.encoder.hidden_size + self.decoder.hidden_size, self.target_vocab_size)

    def forward(self, source, target, teacher_forcing=0.5):
        # source: [batch, seq_length]
        # target: [batch, seq_length]
        batch_size = target.shape[0]
        target_seq_length = target.shape[1]

        encoder_output, hidden = self.encoder(source, self.embedding)
        decoder_input = torch.tensor([special.index("<bos>")] * batch_size).long()

        attention_outputs = torch.zeros(batch_size, target_seq_length, self.target_vocab_size)
        for t in range(1, target_seq_length):
            decoder_output, hidden = self.decoder(decoder_input, hidden, self.embedding)
            # encoder output, decoder output 두 값을 이용하여 지금 decoding 할 context 생성
            # decoder output, context 이용하여 attention 적용된 output 도출
            # attention output 사용하여 greedy decoding
            context = self.attention(encoder_output, decoder_output)
            attention_output = self.linear(torch.cat([decoder_output, context], dim=2).squeeze(1))
            attention_outputs[:, t, :] = attention_output
            teacher = target[:, t]
            top1 = attention_output.argmax(1)
            decoder_input = teacher if random.random() < teacher_forcing else top1
        return attention_outputs


embedding_dim = 32
hidden_dim = 32
enc = Encoder(tokenizer.get_vocab_size(), embedding_dim, hidden_dim)
dec = Decoder(tokenizer.get_vocab_size(), embedding_dim, hidden_dim)
att = Attention()
seq2seq_att = Seq2SeqAttention(enc, dec, att)

decode_test = torch.tensor([[special.index("<bos>")] + [special.index("<pad>")] * (max_seq_length - 1)]).long()

if train_seq2seq_attention:
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(seq2seq_att.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=special.index("<pad>"))

    for epoch in range(700):
        seq2seq_att.train()
        epoch_loss = 0.0
        for batch_source, batch_target, batch_length in chat_data_loader:
            optimizer.zero_grad()
            seq2seq_attention_output = seq2seq_att(batch_source, batch_target)

            seq2seq_attention_output_dim = seq2seq_attention_output.shape[-1]
            seq2seq_attention_output_drop = seq2seq_attention_output[:, 1:, :].reshape(-1, seq2seq_attention_output_dim)
            batch_target_drop = batch_target[:, 1:].reshape(-1)
            loss = criterion(seq2seq_attention_output_drop, batch_target_drop)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / batch_length

        if epoch % 10 == 0:
            print(f"{epoch} epoch loss: {epoch_loss:.4f} / ppl: {math.exp(epoch_loss):.4f}")
            seq2seq_att.eval()
            test = "썸 타는 것도 귀찮아."
            test_token = tokenizer.encode(test)
            test_tensor = torch.tensor(test_token.ids).long().unsqueeze(0)
            test_output = seq2seq_att(test_tensor, decode_test, 0.0)[:, 1:, :].squeeze(0).argmax(1).detach().tolist()
            recover_test_output = tokenizer.decode(test_output)
            print(recover_test_output.split("<eos>")[0])

    torch.save(seq2seq_att.state_dict(), "checkpoint/seq2seq_attention.pt")

seq2seq_att.load_state_dict(torch.load("checkpoint/seq2seq_attention.pt"))
seq2seq_att.eval()
test = "죽을거 같네"
test_token = tokenizer.encode(test)
test_tensor = torch.tensor(test_token.ids).long().unsqueeze(0)
test_output = seq2seq_att(test_tensor, decode_test, 0.0)[:, 1:, :].squeeze(0).argmax(1).detach().tolist()
recover_test_output = tokenizer.decode(test_output)
print(recover_test_output.split("<eos>")[0])
