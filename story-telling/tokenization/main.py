import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
import tiktoken

# 1. CONFIGURATION
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BLOCK_SIZE = 128   # Context window
BATCH_SIZE = 32    # Sequences per batch
MAX_ITERS = 1000   # Total batches to train before finishing
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

# 2. OPTIMIZED MODEL ARCHITECTURE
class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)   # (B, T, head_size)
        Q = self.query(x) # (B, T, head_size)
        V = self.value(x) # (B, T, head_size)

        # Compute attention scores. K.shape[-1] is head_size
        scores = Q @ K.transpose(-2, -1) * (K.shape[-1] ** -0.5) # (B, T, T)

        # Apply the causal mask to ensure that attention is only applied to the left in the input sequence
        mask = self.tril[:T, :T]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        
        out = weights @ V # (B, T, head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Concatenate head outputs
        out = self.dropout(self.proj(out))
        return out
    
# 3. FEEDFORWARD NETWORK
class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, n_embd, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, d_ff),
            nn.ReLU(),
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
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, 4*n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connection around self-attention
        x = x + self.ffwd(self.ln2(x)) # Residual connection around feed-forward
        return x
    
# 5. THE COMPLETE GPT MODEL 
class NanoStoryGPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.block_size = block_size
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token and position embeddings
        token_emb = self.token_embedding_table(idx)                                 # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = token_emb + pos_emb                                                     # (B, T, n_embd)

        x = self.blocks(x)       # (B, T, n_embd)
        x = self.ln_f(x)         # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Generate new tokens from the model given a context """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] # Crop context to block size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # Focus on last time step
            probs = F.softmax(logits, dim=-1) # Convert to probabilities
            next_id = torch.multinomial(probs, num_samples=1) # Sample
            idx = torch.cat((idx, next_id), dim=1) # Append sampled token
        return idx
    
# 6. GPTDataset
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
    # Use tiktoken metadata
    enc = tiktoken.get_encoding("cl100k_base")
    vocab_size = enc.n_vocab 
    actual_dtype = np.uint32 # tiktoken IDs usually exceed uint16

    if not os.path.exists(bin_path):
        print("Binary data not found. Please run tokens.py first.")
        return

    # 7.2 Create Dataset and DataLoader
    train_ds = GPTDataset(bin_path, actual_dtype, BLOCK_SIZE)
    # Note: increased num_workers for faster loading of large BPE IDs
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 7.3. Init Model & Optimizer
    # The model now uses the large vocab_size (e.g., 199999)
    model = NanoStoryGPTModel(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT).to(device)
    
    # 7.4 Load existing model if available to continue training
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(device) if device == 'cuda' else None

    # 8. TRAINING LOOP
    iter_num = 0
    model.train()

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
                torch.save(model.state_dict(), model_path)
            
            iter_num += 1

    # 9. SAVE THE MODEL
    torch.save(model.state_dict(), model_path)
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()