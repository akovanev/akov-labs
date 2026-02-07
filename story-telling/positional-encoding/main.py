import torch
import torch.nn as nn
import torch.optim as optim
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        # Create a matrix of shape (max_len, d_model) filled with zeros.
        # The rows are positions (0, 1, 2...) and the columns are the embedding dimensions.
        pe = torch.zeros(max_len, d_model)
        
        # 'Represents the absolute index (0, 1, 2...) — the "address" of each word
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # 'div_term' calculates the frequencies for the sine and cosine waves
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # Fill even columns with sine, odd columns with cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # register_buffer ensures pe is saved with the model but not trained (no gradients)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # PositionAwareEmbedding = WordEmbedding + PositionalSignal
        return x + self.pe[:, :x.size(1)]

class SimpleMLP(nn.Module):
    def __init__(self, vocab_size=20, seq_len=10, d_model=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, seq_len)
        
        # We flatten, but we make d_model small so the PE signal is strong
        self.mlp = nn.Sequential(
            nn.Linear(seq_len * d_model, 256),
            nn.GELU(),
            nn.Linear(256, vocab_size) # Predict the VALUE of the neighbor
        )
    
    def forward(self, x, use_pe=True):
        x = self.embed(x)
        if use_pe:
            x = self.pos_enc(x)
        x = x.reshape(x.size(0), -1) 
        return self.mlp(x)

def generate_neighbor_task(batch_size, seq_len=10, vocab_size=20):
    """
    Task: Find the token '2'. The Target is the value of the token 
    immediately to its RIGHT. (If 2 is at the end, target is the first token).
    """
    seqs = torch.randint(3, vocab_size, (batch_size, seq_len))
    trigger_pos = torch.randint(0, seq_len, (batch_size,))
    
    targets = []
    for i in range(batch_size):
        seqs[i, trigger_pos[i]] = 2 # The "Marker"
        # The target is the value at the next position (circular)
        neighbor_idx = (trigger_pos[i] + 1) % seq_len
        targets.append(seqs[i, neighbor_idx])
        
    return seqs, torch.tensor(targets)

# Model Initialization
model = SimpleMLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training Loop
print("Training WITH Positional Encoding...")
for epoch in range(1001):
    seqs, targets = generate_neighbor_task(64)
    logits = model(seqs, use_pe=True)
    loss = criterion(logits, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        acc = (logits.argmax(-1) == targets).float().mean().item()
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.4f}")

print("Training complete!")
# Print weights of the very first Linear layer in the MLP
mlp_weights = model.mlp[0].weight.data
print(f"MLP Layer 1 Weights (Mean): {mlp_weights.mean():.6f}")
print(f"MLP Layer 1 Weights (Std):  {mlp_weights.std():.6f}")

# Show actual vs expected
def evaluate_and_compare(model, num_samples=10):
    model.eval() # Set to evaluation mode
    seqs, targets = generate_neighbor_task(num_samples)
    
    print(f"\n{'='*20} SAMPLE RESULTS {'='*42}")
    print(f"{'Input Sequence':<45} | {'Target':<8} | {'Predicted':<10} | {'Status'}")
    print("-" * 78)
    
    with torch.no_grad():
        # Get predictions with PE
        logits = model(seqs, use_pe=True)
        preds = logits.argmax(-1)
        
        for i in range(num_samples):
            seq_str = str(seqs[i].tolist())
            target_val = targets[i].item()
            pred_val = preds[i].item()
            status = "✅" if target_val == pred_val else "❌"
            
            print(f"{seq_str:<45} | {target_val:<8} | {pred_val:<10} | {status}")
    print("="*78)

evaluate_and_compare(model)