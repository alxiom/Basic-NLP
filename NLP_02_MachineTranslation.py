import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as ftn
from torch.utils.data import Dataset, DataLoader
from tokenizers import CharBPETokenizer
from bokeh.layouts import column
from bokeh.plotting import figure, show

# Created by Alex Kim
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
special = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<cls>", "<mask>"]

sample_chat = False
train_tokenizer = False
show_analysis = False
train_seq2seq = True

if sample_chat:
    samples = 128
    train_data = pd.read_csv("data/chat.csv", header=0).sample(samples).reset_index(drop=True).drop(columns=["label"])
    train_data.to_csv("data/chat_sample.csv", index=False)
    with open("data/chat_sample.txt", "w", encoding="utf-8") as f:
        for i in range(len(train_data)):
            row = train_data.loc[i]
            query = row["Q"]
            answer = row["A"]
            f.write(f"{query} {answer}\n")

# chat sample text --> vocab.json, merges.txt
if train_tokenizer:
    tokenizer = CharBPETokenizer()
    tokenizer.train(files=["data/chat_sample.txt"], vocab_size=1500, special_tokens=special, min_frequency=1)
    tokenizer.save_model(f"data")

tokenizer = CharBPETokenizer(vocab="data/vocab.json", merges="data/merges.txt")
tokenize_sample_text = tokenizer.encode("인연이 있다고 생각해?")
print(tokenize_sample_text.tokens)
print("--")

recover = tokenizer.decode(tokenize_sample_text.ids)
print(recover)
print("--")

train_data = pd.read_csv("data/chat_sample.csv", header=0)
print(train_data.head(5))
print(len(train_data))
print("--")

query_tokens = []
answer_tokens = []
query_lengths = {}
answer_lengths = {}
for i in range(len(train_data)):
    row = train_data.loc[i]
    query = row["Q"]
    answer = row["A"]

    tokenize_query = tokenizer.encode(query)
    tokenize_answer = tokenizer.encode(answer)

    query_tokens.append(tokenize_query.ids)
    answer_tokens.append(tokenize_answer.ids)

    query_length = len(tokenize_query.ids)
    answer_length = len(tokenize_answer.ids)

    if query_length in query_lengths:
        query_lengths[query_length] += 1
    else:
        query_lengths[query_length] = 1

    if answer_length in answer_lengths:
        answer_lengths[answer_length] += 1
    else:
        answer_lengths[answer_length] = 1

if show_analysis:
    sample_data = train_data.loc[99]
    print(sample_data)
    print("--")

    sample_query = sample_data["Q"]
    print(sample_query)
    print("--")

    tokenize_sample_query = tokenizer.encode(sample_query)
    print(tokenize_sample_query.tokens)
    print(tokenize_sample_query.ids)
    print("--")

    x_axis = list(range(1, max(max(query_lengths), max(answer_lengths)) + 1))
    query_length_list = [query_lengths.get(i, 0) for i in x_axis]
    answer_length_list = [answer_lengths.get(i, 0) for i in x_axis]
    x_axis = [str(i) for i in x_axis]

    plot_query = figure(title="query dist.", x_range=x_axis, plot_height=250, toolbar_location=None, tools="")
    plot_query.vbar(x=x_axis, top=query_length_list, width=0.9)
    plot_query.xgrid.grid_line_color = None
    plot_query.y_range.start = 0

    plot_answer = figure(title="answer dist.", x_range=x_axis, plot_height=250, toolbar_location=None, tools="")
    plot_answer.vbar(x=x_axis, top=answer_length_list, width=0.9)
    plot_answer.xgrid.grid_line_color = None
    plot_answer.y_range.start = 0

    show(column(plot_query, plot_answer))


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
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden, embedding):
        # x: [batch] --> need second dimension as 1
        # hidden: [encoder_layers = 1, batch, hidden_dim]
        x = x.unsqueeze(1)
        x = embedding(x)
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x.squeeze(1))
        return x, hidden


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = nn.Embedding(self.encoder.input_size, self.encoder.embedding_size)
        self.target_vocab_size = self.decoder.output_size

    def forward(self, source, target, teacher_forcing=0.5):
        # source: [batch, seq_length]
        # target: [batch, seq_length]
        batch_size = target.shape[0]
        target_seq_length = target.shape[1]

        _, hidden = self.encoder(source, self.embedding)
        decoder_input = torch.tensor([special.index("<bos>")] * batch_size).long()

        decoder_outputs = torch.zeros(batch_size, target_seq_length, self.target_vocab_size)
        for t in range(1, target_seq_length):
            decoder_output, hidden = self.decoder(decoder_input, hidden, self.embedding)
            decoder_outputs[:, t, :] = decoder_output
            teacher = target[:, t]
            top1 = decoder_output.argmax(1)
            decoder_input = teacher if random.random() < teacher_forcing else top1
        return decoder_outputs


embedding_dim = 128
hidden_dim = 128
enc = Encoder(tokenizer.get_vocab_size(), embedding_dim, hidden_dim)
dec = Decoder(tokenizer.get_vocab_size(), embedding_dim, hidden_dim)
seq2seq = Seq2Seq(enc, dec)

decode_test = torch.tensor([[special.index("<bos>")] + [special.index("<pad>")] * (max_seq_length - 1)]).long()

if train_seq2seq:
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=special.index("<pad>"))

    for epoch in range(300):
        seq2seq.train()
        epoch_loss = 0.0
        for batch_source, batch_target, batch_length in chat_data_loader:
            optimizer.zero_grad()
            seq2seq_output = seq2seq(batch_source, batch_target)

            seq2seq_output_dim = seq2seq_output.shape[-1]
            seq2seq_output_drop = seq2seq_output[:, 1:, :].reshape(-1, seq2seq_output_dim)
            batch_target_drop = batch_target[:, 1:].reshape(-1)
            loss = criterion(seq2seq_output_drop, batch_target_drop)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / batch_length

        if epoch % 10 == 0:
            print(f"{epoch} epoch loss: {epoch_loss:.4f} / ppl: {math.exp(epoch_loss):.4f}")
            seq2seq.eval()
            test = "썸 타는 것도 귀찮아."
            test_token = tokenizer.encode(test)
            test_tensor = torch.tensor(test_token.ids).long().unsqueeze(0)
            test_output = seq2seq(test_tensor, decode_test, 0.0)[:, 1:, :].squeeze(0).argmax(1).detach().tolist()
            recover_test_output = tokenizer.decode(test_output)
            print(recover_test_output.split("<eos>")[0])

    torch.save(seq2seq.state_dict(), "checkpoint/seq2seq.pt")

seq2seq.load_state_dict(torch.load("checkpoint/seq2seq.pt"))
seq2seq.eval()
test = "죽을거 같네"
test_token = tokenizer.encode(test)
test_tensor = torch.tensor(test_token.ids).long().unsqueeze(0)
test_output = seq2seq(test_tensor, decode_test, 0.0)[:, 1:, :].squeeze(0).argmax(1).detach().tolist()
recover_test_output = tokenizer.decode(test_output)
print(recover_test_output.split("<eos>")[0])
