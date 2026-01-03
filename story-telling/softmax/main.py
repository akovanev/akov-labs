import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

data_dir = os.path.join(os.path.dirname(__file__), 'data')
train_path = os.path.join(data_dir, 'mnist_train.csv')
test_path = os.path.join(data_dir, 'mnist_test.csv')

# 1. Load YOUR CSV files (already split!)
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Extract features/labels (assuming standard format: label + 784 pixels)
X_train = train_data.drop('label', axis=1).values / 255.0
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values / 255.0  
y_test = test_data['label'].values

# 2. PyTorch tensors
train_dataset = TensorDataset(
    torch.FloatTensor(X_train), 
    torch.LongTensor(y_train)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test), 
    torch.LongTensor(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# 3. Softmax model (unchanged)
class SoftmaxRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)
    
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)

model = SoftmaxRegression()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Training
for epoch in range(21):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. Test accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 6. Test first image

with torch.no_grad():
    first_img = torch.FloatTensor(X_test[0:1])  # Shape: (1, 784)
    pred_probs = model(first_img)               # Softmax output
    _, predicted = torch.max(pred_probs, 1)     # Predicted class
    true_label = y_test[0]                      # Ground truth

# Display image + prediction
img = X_test[0].reshape(28, 28)  # Your reshape (needs no tensor conversion)
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
probs_list = [f"{p:.4f}" for p in pred_probs[0].numpy()]
probs_str = " ".join(probs_list[:10])
plt.title(f"True: {int(true_label)}\nPredicted: {predicted.item()}\nProbs: {probs_str}")
plt.axis('off')
plt.show()
