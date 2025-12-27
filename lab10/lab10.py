import torch
import torch.nn as nn
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
class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model) # [6 -> 6]
        self.W_K = nn.Linear(d_model, d_model) # [6 -> 6]
        self.W_V = nn.Linear(d_model, d_model) # [6 -> 6]
        self.W_O = nn.Linear(d_model, d_model) # [6 -> 6]

    def forward(self, x):
        B, T, _ = x.shape
        # .view() splits into heads (B, T, n_heads, d_k)
        # .transpose() brings heads forward (B, n_heads, T, d_k)
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        # contiguous().view() to merge heads back
        attn = (weights @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(attn), weights

# 4. TRAINING SETUP
attn = SimpleMultiHeadAttention(d_model, n_heads)
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