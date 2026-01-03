import numpy as np
import torch
import torch.nn as nn
from plot import plot_loss  
from plot import plot_training  

# Configuration
epochs = 2000 # Number of training epochs
learning_rate = 1e-2 # Learning rate for weight updates

# Data
X = np.array([
    [0.1, 0.05],  # ham: low spam words, low caps
    [0.2, 0.1],   # ham
    [0.15, 0.08], # ham
    [0.9, 0.7],   # spam: high "free"/"win" words, ALL CAPS
    [0.85, 0.65], # spam
    [0.95, 0.75], # spam
    [0.3, 0.15],   # ham
    [0.88, 0.68], # spam
    [0.93, 0.1],  # spam
    [0.2, 0.92],  # spam
    [0.03, 0.3],  # ham
    [0.45, 0.65]  # spam
])
y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1])
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# Model
class SigmoidPerceptron(nn.Module):
    def __init__(self, input_features):
        super(SigmoidPerceptron, self).__init__()
        self.linear = nn.Linear(input_features, 1)  # w · x + b
        self.sigmoid = nn.Sigmoid()                 # σ(z)
    
    def forward(self, x):
        z = self.linear(x)           # Linear combination
        return self.sigmoid(z)       # Sigmoid activation [0,1]
    
    def predict(self, X, threshold=0.5):
        self.eval()
        with torch.no_grad():
            probs = self(X)
            return (probs > threshold).float().squeeze()

# Initialize model
model = SigmoidPerceptron(2)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Training
loss_history = []  # Track loss per epoch
for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    # Average loss for epoch
    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# Test
p1=[0.15, 0.32]
p2=[0.65, 0.15]
p1_tensor = torch.tensor([p1], dtype=torch.float32)  # shape: [1, 2]
p2_tensor = torch.tensor([p2], dtype=torch.float32)  # shape: [1, 2]
pred1 = model.predict(p1_tensor)
pred2 = model.predict(p2_tensor)
print(f"{pred1:.4f} -> {pred1.round()}") # Expected: 0 (ham)
print(f"{pred2:.4f} -> {pred2.round()}") # Expected: 1 (spam)

# Plot
w = model.linear.weight.detach().numpy().flatten()  # shape: (num_features,)
b = model.linear.bias.detach().item() 
plot_loss(epochs, loss_history)
plot_training(X, y, w, b, p1, p2)