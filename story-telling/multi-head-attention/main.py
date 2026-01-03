import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from plot import plot_attention_trajectories

# 1. DATA SETUP
torch.manual_seed(42)
vocab = {
    "rock":   torch.tensor([1.0, 1.0, 1.0]), # Ambiguous center
    "quartz": torch.tensor([1.0, 0.0, 0.0]), # Geology (X)
    "jazz":   torch.tensor([0.0, 1.0, 0.0]), # Music (Y)
    "cliff":  torch.tensor([0.0, 0.0, 1.0]), # Elevation (Z)
}

training_data = [
    (["rock", "quartz"], torch.tensor([1.0, 0.0, 0.2])), 
    (["rock", "jazz"],  torch.tensor([0.1, 1.0, 0.1])),
    (["rock", "cliff"],  torch.tensor([0.5, 0.0, 1.0])),
]

# 2. CONFIGURATION
d_in = 3
d_model = 6
n_heads = 2
d_k = d_model // n_heads # 3 [6 -> 3]
encoder = nn.Linear(d_in, d_model) # [3 -> 6]
decoder = nn.Linear(d_model, d_in) # [6 -> 3]

# 3. MULTI-HEAD ATTENTION MODULE
class Head(nn.Module):
    """ A single head of self-attention """
    def __init__(self, d_model, head_size):
        super().__init__()
        # Each head has its own smaller projection matrices
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)   # (B, T, head_size)
        Q = self.query(x) # (B, T, head_size)
        V = self.value(x) # (B, T, head_size)

        # Compute attention scores
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        scores = Q @ K.transpose(-2, -1) * (self.head_size ** -0.5)
        weights = F.softmax(scores, dim=-1)
        
        # Perform the weighted aggregation of the values
        out = weights @ V # (B, T, head_size)
        return out, weights

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, d_model, n_heads):
        super().__init__()
        head_size = d_model // n_heads
        # Create a list of independent Head modules
        self.heads = nn.ModuleList([Head(d_model, head_size) for _ in range(n_heads)])
        # Final projection to bring it back to d_model size
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Run each head and concatenate the results along the feature dimension
        # Each head returns (out, weights)
        head_outputs = [h(x) for h in self.heads]
        
        # Concatenate the 'out' parts: (B, T, head_size * n_heads)
        out = torch.cat([res[0] for res in head_outputs], dim=-1)
        
        # Collect weights for visualization/analysis if needed
        weights = torch.stack([res[1] for res in head_outputs], dim=1)
        
        return self.proj(out), weights

# 4. TRAINING SETUP
attn = MultiHeadAttention(d_model, n_heads)
# Combine all model parameters for optimization
params = list(encoder.parameters()) + list(attn.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=0.01)
criterion = nn.MSELoss()

# 5. THE TRAINING LOOP
print("Training the Attention Lenses...")
for epoch in range(301):
    epoch_loss = 0
    for words, target_meaning in training_data:
        optimizer.zero_grad()
        
        x = torch.stack([vocab[w] for w in words]).unsqueeze(0)  # Add batch dimension [1, 2, 3]
       
        x_enc = encoder(x)  # (1, 2, 6)
        out, _ = attn(x_enc)  # (1, 2, 6)
        rock_out = decoder(out)[0, 0, :]
        
        loss = criterion(rock_out, target_meaning)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Avg Loss: {epoch_loss/3:.4f}")

print("Training complete!") 

# 5. FINAL TEST FUNCTION
def get_rock_vec(context_word):
    with torch.no_grad():
        x = torch.stack([vocab["rock"], vocab[context_word]]).unsqueeze(0)
        x_enc = encoder(x)
        out, weights = attn(x_enc)
        rock_vec = decoder(out)[0, 0, :]
        return rock_vec.numpy()

rock_orig = vocab["rock"].numpy()
rock_q = get_rock_vec("quartz")
rock_m = get_rock_vec("jazz")
rock_c = get_rock_vec("cliff")

print("\n" + "="*40)
print("RESULTS: 'ROCK' VECTOR AFTER ATTENTION")
print("="*40)
print(f"Original Rock:    {['1.000', '1.000', '1.000']}")
print(f"Rock + Quartz:    {[f'{val:.3f}' for val in rock_q]}   (Geology focus)")
print(f"Rock + Jazz:      {[f'{val:.3f}' for val in rock_m]}   (Music focus)")
print(f"Rock + Cliff:     {[f'{val:.3f}' for val in rock_c]}   (Elevation focus)")

plot_attention_trajectories(vocab, rock_orig, rock_q, rock_m, rock_c)