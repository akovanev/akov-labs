import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate synthetic training data (y = 2x1 + 3x2 + noise)
np.random.seed(42)
size = 10000
X = np.random.randn(size, 2).astype(np.float32) * 2
y = (2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(size) * 0.5).reshape(-1, 1).astype(np.float32)

dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the architecture factory to ensure fresh weights for each run
# 3-layer feedforward neural network 
def create_model():
    return nn.Sequential(
        nn.Linear(2, 64),  # 2 inputs → 64 neurons (2×64 + 64 = 192 params)
        nn.ReLU(),         # Adds non-linearity (0 for negative, identity for positive)
        nn.Linear(64, 32), # 64 inputs → 32 neurons (64×32 + 32 = 2,112 params)
        nn.ReLU(),         # Adds non-linearity (0 for negative, identity for positive)
        nn.Linear(32, 1)   # 32 inputs → 1 output (32×1 + 1 = 33 params)
    ).to(device)

criterion = nn.MSELoss()

# Define optimizer configurations properly
optimizer_configs = [
    ('SGD', optim.SGD, {'lr': 0.01}),
    ('SGD+Momentum', optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
    ('Adam', optim.Adam, {'lr': 0.001}),
    ('AdamW', optim.AdamW, {'lr': 0.001, 'weight_decay': 0.01})
]

for name, opt_class, kwargs in optimizer_configs:
    # 1. Reset model for each optimizer to ensure a fair comparison
    model_copy = create_model()
    
    # 2. Instantiate optimizer with the specific model's parameters
    optimizer = opt_class(model_copy.parameters(), **kwargs)
    
    losses = []
    for epoch in range(50):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model_copy(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_loader))
    
    print(f"{name:15}: Final Loss = {losses[-1]:.4f}")