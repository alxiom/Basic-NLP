import gensim
import konlpy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from bokeh.plotting import figure, show
from tqdm import tqdm

# Created by Alex Kim
torch.manual_seed(42)

text = """
수학은 수식이 복잡해서 어렵다
수학은 공식이 많아서 어렵다
수학은 수식이 이해되면 쉽다
수학은 공식이 능통하면 쉽다
영어는 단어가 많아서 어렵다
영어는 듣기가 복잡해서 어렵다
영어는 단어가 이해되면 쉽다
영어는 듣기가 능통하면 쉽다
국어는 지문이 복잡해서 어렵다
국어는 한문이 많아서 어렵다
국어는 지문이 이해되면 쉽다
국어는 한문이 능통하면 쉽다
""".strip()

words = list(set(text.split()))

words.insert(0, "[PAD]")  # 공백 문자열 (0)
words.insert(1, "[UNK]")  # 알 수 없는 문자열 (1)
print(f"words: {words}")
print("--")

n_word = len(words)
print(f"unique word count: {n_word}")
print("--")

word_to_id = {}
for i, word in enumerate(words):
    word_to_id[word] = i
print(word_to_id)
print("--")

id_to_word = {value: key for key, value in word_to_id.items()}
print(id_to_word)
print("--")

sentences = text.split("\n")
print(sentences)
print("--")

sentences_tokens = []
for sentence in sentences:
    sentences_tokens.append(sentence.split())
print(sentences_tokens)
print("--")

window_size = 2
word_pair_list = []
for sentence_tokens in sentences_tokens:
    for i in range(len(sentence_tokens)):
        window_left_end = max(0, i - window_size)
        window_right_end = min(len(sentence_tokens) - 1, i + window_size)
        center = sentence_tokens[i]
        outer = [sentence_tokens[j] for j in range(window_left_end, window_right_end + 1) if j != i]
        word_pair_list.append({"c": center, "o": outer})
print(len(word_pair_list))
print(word_pair_list)
print("--")


class LoadDataset(Dataset):

    def __init__(self, x_data, y_data):
        super(LoadDataset, self).__init__()
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return len(self.y_data)


def draw_figure(xy, word_dict, title):
    p = figure(tools="save", title=title)
    p.plot_width = 800
    p.plot_height = 800
    p.outline_line_color = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.axis.major_label_standoff = 0

    p.circle([x for x, y in xy], [y for x, y in xy], fill_color="white", line_color="black", line_width=1, size=12)
    p.text([x for x, y in xy], [y for x, y in xy], [word_dict[j] for j in range(len(xy))])
    show(p, browser="firefox")


embedding_dim = 2


# skip-gram
class SkipGram(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x

    def get_embedding(self, x):
        return self.embedding(x)


skip_gram_tokens = []
skip_gram_labels = []
for word_pair in word_pair_list:
    center_token = word_pair["c"]
    outer = word_pair["o"]
    for outer_token in outer:
        # input: center token --> label: outer token
        skip_gram_tokens.append(center_token)
        skip_gram_labels.append(outer_token)
print(f"skip-gram tokens: {skip_gram_tokens}")
print(f"skip-gram labels: {skip_gram_labels}")
print("--")

skip_gram_token_ids = np.array([word_to_id[token] for token in skip_gram_tokens])
skip_gram_label_ids = np.array([word_to_id[label] for label in skip_gram_labels])
print(f"skip-gram token_ids: {skip_gram_token_ids}")
print(f"skip-gram label_ids: {skip_gram_label_ids}")
print("--")

skip_gram_dataset = LoadDataset(skip_gram_token_ids, skip_gram_label_ids)
skip_gram_data_loader = DataLoader(skip_gram_dataset, batch_size=512)

skip_gram_model = SkipGram(n_word, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(skip_gram_model.parameters(), lr=1e-4)

for iteration in range(50):
    skip_gram_model.train()
    for epoch in range(500):
        epoch_loss = 0.0
        for token, label in skip_gram_data_loader:
            optimizer.zero_grad()
            prediction = skip_gram_model(token)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"iter: {iteration} / epoch: {epoch} / loss: {epoch_loss}")

    skip_gram_model.eval()
    embedding_vectors = []
    for token_id in range(n_word):
        embedding_vector = skip_gram_model.get_embedding(torch.tensor([token_id]).long())
        embedding_vectors.append(embedding_vector.detach().tolist()[0])
    draw_figure(embedding_vectors, id_to_word, "skip-gram")


# CBOW
class CBOW(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        return x

    def get_embedding(self, x):
        return self.embedding(x)


cbow_tokens = []
cbow_labels = []
for word_pair in word_pair_list:
    c = word_pair["c"]
    o = word_pair["o"]
    o += ["[PAD]"] * (window_size * 2 - len(o))  # outer token < window_size * 2 --> add [PAD]
    # input: outer token --> label: center token
    cbow_tokens.append(o)
    cbow_labels.append(c)
print(f"tokens : {cbow_tokens}")
print(f"labels : {cbow_labels}")
print("--")

cbow_token_ids = np.array([[word_to_id[t] for t in token] for token in cbow_tokens])
cbow_label_ids = np.array([word_to_id[label] for label in cbow_labels])
print(f"cbow token_ids: {cbow_token_ids}")
print(f"cbow label_ids: {cbow_label_ids}")
print("--")

cbow_dataset = LoadDataset(cbow_token_ids, cbow_label_ids)
cbow_data_loader = DataLoader(cbow_dataset, batch_size=512)

cbow_model = CBOW(n_word, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cbow_model.parameters(), lr=1e-4)

for iteration in range(50):
    cbow_model.train()
    for epoch in range(500):
        epoch_loss = 0.0
        for token, label in cbow_data_loader:
            optimizer.zero_grad()
            prediction = cbow_model(token)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"iter: {iteration} / epoch: {epoch} / loss: {epoch_loss}")

    cbow_model.eval()
    embedding_vectors = []
    for token_id in range(n_word):
        embedding_vector = cbow_model.get_embedding(torch.tensor([token_id]).long())
        embedding_vectors.append(embedding_vector.detach().tolist()[0])
    draw_figure(embedding_vectors, id_to_word, "CBOW")


# Gensim: word2vec
word2vec = gensim.models.Word2Vec(size=embedding_dim, window=2, min_count=1)  # default CBOW
word2vec.build_vocab(sentences=sentences_tokens)
word2vec.train(sentences=sentences_tokens, total_examples=len(sentences_tokens), epochs=30000)

similar = word2vec.wv.most_similar(u"국어는")
print(similar)
print("--")

wv_vocab_list = list(word2vec.wv.vocab)
wv_vocab_dict = {i: token for i, token in enumerate(wv_vocab_list)}
draw_figure(list(word2vec.wv[wv_vocab_list]), wv_vocab_dict, "gensim: w2v")

# Gensim: FastText
fasttext = gensim.models.FastText(size=embedding_dim, window=2, min_count=1)
fasttext.build_vocab(sentences=sentences_tokens)
fasttext.train(sentences=sentences_tokens, total_examples=len(sentences_tokens), epochs=30000)

similar = fasttext.wv.most_similar(u"국어는")
print(similar)
print("--")

ft_vocab_list = list(fasttext.wv.vocab)
ft_vocab_dict = {i: token for i, token in enumerate(ft_vocab_list)}
draw_figure(list(fasttext.wv[ft_vocab_list]), ft_vocab_dict, "gensim: fasttext")

# NSMC + konlpy + gensim
nsmc_data = pd.read_csv("data/ratings.txt", header=0, delimiter="\t", quoting=3)
print(f"data count: {len(nsmc_data)}")
print("--")

nsmc_data["document"] = nsmc_data["document"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
print(f"data count (korean filtered): {len(nsmc_data)}")
print("--")

nsmc_data = nsmc_data.dropna(how="any")
print(f"data count (null filtered): {len(nsmc_data)}")
print("--")

stopwords = ["의", "가", "이", "은", "들", "는", "좀", "잘", "걍", "과", "도", "를", "으로", "자", "에", "와", "한", "하다"]

okt = konlpy.tag.Okt()
print(okt.morphs("아버지가방에들어가신다", stem=True))
print("--")

nsmc_tokens = []
for i, document in enumerate(tqdm(nsmc_data["document"], total=len(nsmc_data))):
    sentence = okt.morphs(document, stem=True)
    sentence = [token for token in sentence if token not in stopwords]
    nsmc_tokens.append(sentence)

word2vec_200 = gensim.models.Word2Vec(sentences=nsmc_tokens, size=200, window=5, min_count=5)
nsmc_vocab_list = list(word2vec_200.wv.vocab)
similar = word2vec_200.wv.most_similar("최민식")
print(similar)
print("--")
