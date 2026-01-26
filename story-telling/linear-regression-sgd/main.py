import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from test import interpret_results

# Data Generation
np.random.seed(42)
square_meters = np.random.randint(100, 300, 200)
bedrooms = np.random.randint(2, 6, 200)
noise_std = 5
noise = np.random.normal(0, noise_std, 200)
prices = 10 + (0.85 * square_meters) + (15 * bedrooms) + noise

# Data
X_data = np.column_stack((square_meters, bedrooms))
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
    epoch_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 400 == 0:
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}, Average loss: {avg_loss:.4f}")

# Test
interpret_results(model, X_data, X_tensor, prices, noise_std)