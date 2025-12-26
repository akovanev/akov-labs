import torch
import torch.nn as nn
import torch.optim as optim
from plot import plot_attention_trajectories

# 1. SET SEED FOR REPRODUCIBILITY
# This ensures the "random" W matrices start the same way every time
torch.manual_seed(42)

# 2. VOCABULARY (3D Concept Space)
# We define anchor words at the tips of the X, Y, and Z axes
vocab = {
    "rock":   torch.tensor([1.0, 1.0, 1.0]), # Ambiguous center
    "quartz": torch.tensor([1.0, 0.0, 0.0]), # Lithology (X)
    "music":  torch.tensor([0.0, 1.0, 0.0]), # Genre (Y)
    "cliff":  torch.tensor([0.0, 0.0, 1.0]), # Elevation (Z)
}

# 3. INITIALIZE "MAGIC" MATRICES (Linear Layers)
# These start as random coefficients but will be "tuned" during training
# d_in: Input size (3 dimensions: Lithology, Music, Elevation)
# d_k:  Internal 'projection' size (the space where Query and Key vectors "meet")
# d_v:  Output size (we keep it same as input for simplicity)
d_in, d_k, d_v = 3, 2, 3
W_Q = nn.Linear(d_in, d_k, bias=False)
W_K = nn.Linear(d_in, d_k, bias=False)
W_V = nn.Linear(d_in, d_v, bias=False)

# 4. TRAINING SETUP
# We use Adam optimizer to update the coefficients in W_Q, W_K, and W_V
optimizer = optim.Adam(list(W_Q.parameters()) + list(W_K.parameters()) + list(W_V.parameters()), lr=0.01)
criterion = nn.MSELoss()

# Training Data: (Context Pair, Target vector for "rock")
training_data = [
    (["rock", "quartz"], torch.tensor([1.0, 0.0, 0.2])), 
    (["rock", "music"],  torch.tensor([0.1, 1.0, 0.1])),
    (["rock", "cliff"],  torch.tensor([0.5, 0.0, 1.0])),
]

# 5. THE TRAINING LOOP (Learning the coefficients)
print("Training the Attention Lenses...")
for epoch in range(301):
    epoch_loss = 0
    for words, target_meaning in training_data:
        optimizer.zero_grad()
        
        # Stack the two words into a sequence matrix [2, 3]
        x = torch.stack([vocab[w] for w in words])
       
        # Step 1: Projection
        Q, K, V = W_Q(x), W_K(x), W_V(x)
        
        # Step 2: Attention Calculation (The Dot Product)
        # We divide by sqrt(d_k) to keep scores stable
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        
        # Step 3: Softmax (Turning scores into weights that sum to 1)
        weights = torch.softmax(scores, dim=-1)
        
        # Step 4: Aggregation (Weighted sum of values)
        # We only care about the first output: the updated "rock"
        rock_output = torch.matmul(weights, V)[0]
        
        # Step 5: Backpropagation
        loss = criterion(rock_output, target_meaning)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Avg Loss: {epoch_loss/3:.4f}")

print("Training complete!") 
print(f"Weights learned:\n{W_Q.weight.data}\n{W_K.weight.data}\n{W_V.weight.data}\n")

# 6. FINAL TEST FUNCTION
def get_rock_vec(context_word):
    x = torch.stack([vocab["rock"], vocab[context_word]])
    with torch.no_grad():
        Q, K, V = W_Q(x), W_K(x), W_V(x)
        weights = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5), dim=-1)
        rock_vec = torch.matmul(weights, V)[0]
        return rock_vec.numpy()

rock_orig = vocab["rock"].numpy()
rock_q = get_rock_vec("quartz")
rock_m = get_rock_vec("music")
rock_c = get_rock_vec("cliff")

print("\n" + "="*40)
print("RESULTS: 'ROCK' VECTOR AFTER ATTENTION")
print("="*40)
print(f"Original Rock:    {['1.000', '1.000', '1.000']}")
# Replace your print line with this:
print(f"Rock + Quartz:    {[f'{val:.3f}' for val in rock_q]}  (Lithology focus)")
print(f"Rock + Music:     {[f'{val:.3f}' for val in rock_m]}   (Genre focus)")
print(f"Rock + Cliff:     {[f'{val:.3f}' for val in rock_c]}   (Elevation focus)")

plot_attention_trajectories(vocab, rock_orig, rock_q, rock_m, rock_c)