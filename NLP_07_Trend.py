import torch
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

kc_bert_sample_text = "배가 바다를 향해 떠난다"

kc_bert_tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
kc_bert_model = BertModel.from_pretrained("beomi/kcbert-base")
kc_bert_inputs = kc_bert_tokenizer(kc_bert_sample_text, return_tensors="pt")
kc_bert_outputs = kc_bert_model(**kc_bert_inputs)[0]  # The last hidden-state is the first element of the output tuple

print(kc_bert_tokenizer.tokenize(kc_bert_sample_text))
print(kc_bert_outputs)  # [CLS] [배가] [바다] [를] [향해] [떠난] [다] [SEP]
print(kc_bert_outputs.shape)  # batch * seq_len * hidden_dim

gpt2_sample_text = "The Manhattan bridge"

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
generate = gpt2_tokenizer.encode(gpt2_sample_text)
context = torch.tensor([generate])
past = None
for _ in range(25):
    output, past = gpt2_model(context, past_key_values=past)
    token = torch.argmax(output[..., -1, :])
    generate += [token.tolist()]
    context = token.unsqueeze(0)

decode_generate = gpt2_tokenizer.decode(generate)
print(decode_generate)
