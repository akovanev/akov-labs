import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Data
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Hidden layer
        self.fc2 = nn.Linear(2, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = MLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training
for epoch in range(10000):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

# Test
with torch.no_grad():
    predictions = model(X_tensor)
    print("Predictions:")

    GREEN = '\033[92m'  # Green for correct
    RED = '\033[91m'    # Red for incorrect
    RESET = '\033[0m'   # Reset color

    for i, pred in enumerate(predictions):
        is_correct = (pred > 0.5).float() == y_tensor[i]
        color = GREEN if is_correct else RED
        print(f"Input: {X[i]}, Predicted: {pred.item():.4f}, Actual: {y[i][0]}", end=' ')
        print(f"{color}✓{RESET}" if is_correct else f"{color}✗{RESET}")

    accuracy = ((predictions > 0.5).float() == y_tensor).float().mean()
    print(f"Accuracy: {accuracy.item()*100:.2f}%")

