import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
from gpt_prep import run_gpt_prep

# 1. CONFIGURATION
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BLOCK_SIZE = 128   # Context window
BATCH_SIZE = 32    # Sequences per batch
MAX_ITERS = 5000   # Max iterations to train before finishing
MIN_ITER_TO_SAVE = 1000
EVAL_INTERVAL = 10
LEARNING_RATE = 3e-4
N_EMBD = 384 # The same as d_model in previous labs
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.1

# Paths
data_dir = os.path.join(os.path.dirname(__file__), 'data')
input_path = os.path.join(data_dir, 'train.csv')
bin_path = os.path.join(data_dir, 'train.bin')
vocab_path = os.path.join(data_dir, 'vocab.pt')
model_path = os.path.join(data_dir, 'gpt_lab_model.pt')

# 2. VECTORIZED ATTENTION (The Efficiency Fix)
class CausalSelfAttention(nn.Module):
    """ Vectorized Multi-Head Causal Self-Attention """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        # Key, Query, Value projections for all heads in one linear layer
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() 

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# 3. FEEDFORWARD NETWORK
class FeedForward(nn.Module):
    def __init__(self, n_embd, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, d_ff),
            nn.GELU(), # GELU is more common in modern GPT architectures
            nn.Linear(d_ff, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
# 4. TRANSFORMER BLOCK
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, 4*n_embd, dropout)
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x

# 5. THE GPT MODEL
class NanoStoryGPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

        # Weight initialization
        self.apply(self._init_weights)
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # (1, t)

        # Forward the transformer
        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 6. DATASET & TRAINING (Standard Logic maintained)
class GPTDataset(Dataset):
    def __init__(self, bin_path, dtype, block_size):
        self.data = np.memmap(bin_path, dtype=dtype, mode='r')
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size - 1
    def __getitem__(self, idx):
        chunk = torch.from_numpy(self.data[idx : idx + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]
# 7. Data Preparation

def main():
    # 7.1 Prep Data
    if not os.path.exists(bin_path):
        vocab_size, actual_dtype = run_gpt_prep(input_path, bin_path, vocab_path)
    else:
        word_to_id = torch.load(vocab_path)
        vocab_size = len(word_to_id)
        actual_dtype = np.uint16 if vocab_size < 65535 else np.uint32

    # 7.2 Create Dataset and DataLoader
    train_ds = GPTDataset(bin_path, actual_dtype, BLOCK_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 7.3. Init Model & Optimizer
    model = NanoStoryGPTModel(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT).to(device)

    # 7.4 Load existing model if available to continue training
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(device) if device == 'cuda' else None

    # 8. TRAINING LOOP
    iter_num = 0
    best_loss = float('inf')
    model.train()

    print(f"Training on {device}...")
    print(f"Training started. Max iterations: {MAX_ITERS}")
    finished = False
    while not finished:
        for x, y in train_loader:
            if iter_num >= MAX_ITERS:
                finished = True
                break
            
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast(device_type=device, enabled=device == 'cuda'):
                logits, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True) 

            # forward pass
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            if iter_num % EVAL_INTERVAL == 0:
                print(f"Step {iter_num} | Loss: {loss.item():.4f}")

            if iter_num > MIN_ITER_TO_SAVE and loss.item() < best_loss:
                print(f"New best loss: {loss.item():.4f} (previous: {best_loss:.4f}). Saving model...")
                best_loss = loss.item()
                torch.save(model.state_dict(), model_path)
            
            iter_num += 1

    # Training complete
    print("Training complete.")

if __name__ == "__main__":
    main()