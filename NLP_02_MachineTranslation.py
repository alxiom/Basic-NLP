# Created by Alex Kim
import glob
import json
import pandas as pd
from tqdm import tqdm
from tokenizers import CharBPETokenizer

# wiki text --> vocab
file_list = []
for file_name in glob.iglob("data/kowiki/*/*", recursive=True):
    file_list.append(file_name)

with open("data/kowiki.txt", "w", encoding="utf-8") as f_write:
    for file_name in tqdm(file_list):
        with open(file_name, "r", encoding="utf-8") as f_read:
            for read_line in f_read:
                read_line = read_line.strip()
                if read_line:
                    text = json.loads(read_line)["text"]
                    refine_text = [t for t in text.split("\n") if len(t) > 0]
                    f_write.write(" ".join(refine_text) + "\n")

data_path = "data/kowiki.txt"
vocab_size = 8000

tokenizer = CharBPETokenizer()
tokenizer.train(
    files=[data_path],
    vocab_size=vocab_size,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]"],
    min_frequency=2,
)

tokenizer.save_model(f"data")
print("saved")

tokenizer = CharBPETokenizer(vocab="data/vocab.json", merges="data/merges.txt")
output = tokenizer.encode("상수란 그 값이 변하지 않는 불변량으로, 변수의 반대말이다.")
print(output.tokens)
recover = tokenizer.decode(output.ids)
print(recover)

train_data = pd.read_csv("data/chat.csv", header=0)
print("--")
print(train_data.head(10))
print("--")
print(train_data.loc[3])
print("--")

