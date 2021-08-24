from transformers import BertTokenizer, BertModel
from transformers import pipeline, set_seed

kc_bert_sample_text = "배가 바다를 향해 떠난다"

kc_bert_tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
kc_bert_model = BertModel.from_pretrained("beomi/kcbert-base")
kc_bert_inputs = kc_bert_tokenizer(kc_bert_sample_text, return_tensors="pt")
kc_bert_outputs = kc_bert_model(**kc_bert_inputs)[0]  # The last hidden-state is the first element of the output tuple

print(kc_bert_tokenizer.tokenize(kc_bert_sample_text))
print(kc_bert_outputs)  # [CLS] [배가] [바다] [를] [향해] [떠난] [다] [SEP]
print(kc_bert_outputs.shape)  # batch * seq_len * hidden_dim

gpt2_sample_text = "The Manhattan bridge"

generator = pipeline("text-generation", model="gpt2")
set_seed(42)
results = generator(gpt2_sample_text, max_length=30, num_return_sequences=5)

for result in results:
    print(result["generated_text"])
    print("--")
