import os
import math
import requests
import tiktoken
import torch
import torch.nn as nn

context_length = 16
d_model = 64
batch_size = 4
num_heads = 4
head_size = d_model // num_heads
num_blocks = 8
learning_rate = 1e-3
dropout = 0.1
max_iters = 100
eval_interval = 50
eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists("Chapter 2/sales_textbook.txt"):
    url = "https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true"
    with open("Chapter 2/sales_textbook.txt", "wb") as f:
        f.write(requests.get(url).content)
    
with open("Chapter 2/sales_textbook.txt", "r") as f:
    text = f.read()

encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device = device)

train_index = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_index]
val_data = tokenized_text[train_index:]

class FeedforwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, head_size)
        self.Wk = nn.Linear(d_model, head_size)
        self.Wv = nn.Linear(d_model, head_size)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal = 1))
    
    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        T = Q.shape[1]  # 实际序列长度，生成时可能小于 context_length

        dot_product = Q @ K.transpose(-2, -1) / math.sqrt(head_size)
        # 切片为 [:T, :T]，避免生成阶段序列短于 context_length 时维度不匹配
        dot_product = dot_product.masked_fill(self.mask[:T, :T] == 1, float('-inf'))
        weights = torch.softmax(dot_product, dim = -1)
        output = weights @ V
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([ScaledDotProductAttention() for _ in range(num_heads)])
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        heads = [head(x) for head in self.heads]
        output = torch.cat(heads, dim = -1)
        output = self.Wo(output)
        output = self.dropout(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_atttention = MultiHeadAttention()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.feedforward_network = FeedforwardNetwork(d_model, d_model * 4)
    
    def forward(self, x):
        x = x + self.multi_head_atttention(self.layer_norm1(x))
        x = x + self.feedforward_network(self.layer_norm2(x))
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(max_token_value + 1, d_model)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock() for _ in range(num_blocks)])
        self.linear = nn.Linear(d_model, max_token_value + 1)
    
    def forward(self, inputs, targets = None):
        B, T = inputs.shape
        position_encoding = torch.zeros(context_length, d_model, device = device)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        position_encoding[:, 0::2] = torch.sin(position / (10000 ** (torch.arange(0, d_model, 2) / d_model)))
        position_encoding[:, 1::2] = torch.cos(position / (10000 ** (torch.arange(0, d_model, 2) / d_model)))
        position_embedding = position_encoding[:T, :].to(device)

        x = self.embedding(inputs) + position_embedding
        x = self.transformer_blocks(x)
        logits = self.linear(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = nn.CrossEntropyLoss()(logits, targets)
        else:
            loss = None
        
        return logits, loss
    
    def generate(self, inputs, max_new_tokens = 100):
        for _ in range(max_new_tokens):
            inputs = inputs[:, -context_length:]
            logits, _ = self(inputs)
            logits = logits[:, -1, :]
            probabilities = torch.softmax(logits, dim = -1)
            next_token = torch.multinomial(probabilities, num_samples = 1)
            inputs = torch.cat([inputs, next_token], dim = 1)
        return inputs


model = Model().to(device)


def get_batch(split):
    data = train_data if split == "train" else val_data
    idxs = torch.randint(0, len(data) - context_length, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in idxs])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in idxs])
    return x.to(device), y.to(device)

@torch.no_grad()
def calculate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
tracked_losses = list()
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = calculate_loss()
        tracked_losses.append(losses)
        print(f"iteration {iter}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")
    
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pth")

model.eval()
start = 'The salesperson'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')
