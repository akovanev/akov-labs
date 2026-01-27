import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from plot import plot_image

data_dir = os.path.join(os.path.dirname(__file__), 'data')
train_path = os.path.join(data_dir, 'mnist_train.csv')
test_path = os.path.join(data_dir, 'mnist_test.csv')

# 1. Load CSV files
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

# 3. Softmax model
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
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss/len(train_loader):.4f}")

# 5. Test accuracy
model.eval()
correct = 0
total = X_test.shape[0]

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 6. Test first image

# Choose any valid index: 0 <= img_index < len(X_test)
img_index = 1  # change this to try different images

with torch.no_grad():
    current_img = torch.FloatTensor(X_test[img_index:img_index+1])  # Shape: (1, 784)
    pred_probs = model(current_img)                                 # Softmax output
    _, predicted = torch.max(pred_probs, 1)                         # Predicted class
    true_label = y_test[img_index]   

plot_image(X_test, img_index, true_label, pred_probs, predicted)
