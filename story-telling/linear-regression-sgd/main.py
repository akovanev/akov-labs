import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Data Generation
np.random.seed(42)
sqft = np.random.randint(1200, 3500, 24)
bedrooms = np.random.randint(2, 6, 24)
noise = np.random.normal(0, 20, 24)
prices = 50 + (0.12 * sqft) + (25 * bedrooms) + noise

# Data
X_data = np.column_stack((sqft, bedrooms))
y_data = prices
X_norm = (X_data - X_data.mean(0)) / X_data.std(0)  # # Normalize features (CRITICAL for stable SGD)
X_tensor = torch.from_numpy(X_norm.astype(np.float32))
y_tensor = torch.from_numpy(y_data.astype(np.float32)).unsqueeze(1)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Start small
criterion = nn.MSELoss()

#Training
for epoch in range(10001):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
    if epoch % 400 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test
with torch.no_grad():
    w_norm = model.weight.squeeze().numpy() # [w1_norm, w2_norm]
    b_norm = model.bias.item()             # scalar bias

# 1. Get stats used during normalization
X_mean = X_data.mean(axis=0)
X_std = X_data.std(axis=0)
# 2. Correct Un-normalization math
beta = w_norm / X_std
b = b_norm - np.sum((w_norm * X_mean) / X_std)

print(f"True betas: [0.12, 25], Intercept: 50")
print(f"Recovered: Beta={beta}, Intercept={b:.4f}")