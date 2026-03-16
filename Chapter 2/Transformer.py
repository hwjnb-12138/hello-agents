import os
import math
import requests
import tiktoken
import torch

context_length = 14
d_model = 64
batch_size = 4
head_num = 4

if not os.path.exists("Chapter 2/sales_textbook.txt"):
    url = "https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true"
    with open("Chapter 2/sales_textbook.txt", "wb") as f:
        f.write(requests.get(url).content)
    
with open("Chapter 2/sales_textbook.txt", "r") as f:
    text = f.read()

encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)

train_index = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_index]
val_data = tokenized_text[train_index:]

idxs = torch.randint(0, len(train_data) - context_length, (batch_size,))
inputs = torch.stack([train_data[i : i + context_length] for i in idxs])
targets = torch.stack([train_data[i + 1 : i + context_length + 1] for i in idxs])

# Embedding
max_token = tokenized_text.max().item() + 1
embedding = torch.nn.Embedding(max_token, d_model)
inputs_embedding = embedding(inputs)
targets_embedding = embedding(targets)

# Position Ecoding
position_encoding = torch.zeros(context_length, d_model)
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
position_encoding[:, 0::2] = torch.sin(position / (10000 ** (torch.arange(0, d_model, 2) / d_model)))
position_encoding[:, 1::2] = torch.cos(position / (10000 ** (torch.arange(0, d_model, 2) / d_model)))
position_encoding = position_encoding.unsqueeze(0).expand(batch_size, -1, -1)

inputs_embedding = inputs_embedding + position_encoding
targets_embedding = targets_embedding + position_encoding

# MultiHead Attention
Wq = nn.Linear(d_model, d_model)
Wk = nn.Linear(d_model, d_model)
Wv = nn.Linear(d_model, d_model)

Q = Wq(inputs_embedding)
K = Wk(inputs_embedding)
V = Wv(inputs_embedding)

Q = Q.view(batch_size, context_length, head_num, d_model // head_num).transpose(1, 2)
K = K.view(batch_size, context_length, head_num, d_model // head_num).transpose(1, 2)
V = V.view(batch_size, context_length, head_num, d_model // head_num).transpose(1, 2)

dot_product = Q @ K.transpose(-2, -1) / math.sqrt(d_model // head_num)
# Mask
mask = torch.triu(torch.ones(context_length, context_length), diagonal = 1).bool()
dot_product = dot_product.masked_fill(mask, float('-inf'))
attention_weight = torch.softmax(dot_product, dim = -1)

A = attention_weight @ V
output = A.transpose(1, 2).view(batch_size, context_length, d_model)
Wo = nn.Linear(d_model, d_model)
output = Wo(output)

# Residual Connection
output += inputs_embedding

# Layer Normalization
layer_norm1 = nn.LayerNorm(d_model)
output = layer_norm1(output)

# Feed Forward
ff = nn.Linear(d_model, d_model * 4)(output)
ff = nn.ReLU()(ff)
ff = nn.Linear(d_model * 4, d_model)(ff)

# Residual Connection
output += ff

# Layer Normalization
layer_norm2 = nn.LayerNorm(d_model)
output = layer_norm2(output)

# Linear Layer
logits = nn.Linear(d_model, max_token)(output)

# Softmax
probabilities = torch.softmax(logits, dim = -1)