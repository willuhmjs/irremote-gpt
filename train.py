import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT
import re
import json
import random
import os
from tokenizer import Tokenizer

# Hyperparameters
batch_size = 16 # Reduced batch size to fix OOM
block_size = 1024
max_iters = 2000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.1

print(f"Using device: {device}")

# --- Data Loading ---

with open('ir_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Build vocab and encode
tok = Tokenizer()
if os.path.exists('vocab.json'):
    print("Loading existing vocab...")
    tok.load_vocab()
    vocab_size = len(tok.stoi)
else:
    print("Building vocab from corpus...")
    vocab_size = tok.build_vocab(text)
    tok.save_vocab()

print(f"Vocabulary size: {vocab_size}")

# Encode entire dataset
data = torch.tensor(tok.encode(text), dtype=torch.long)

# Train/Val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Training ---

model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Final loss report
losses = estimate_loss(model)
print(f"Final step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# Save the model
torch.save(model.state_dict(), 'model.pt')
print("Model saved to model.pt")
