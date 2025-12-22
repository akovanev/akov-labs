import numpy as np
from plot import plot_loss  
from plot import plot_training  

# Configuration
epochs = 2000 # Number of training epochs
learning_rate = 1e-2 # Learning rate for weight updates

#Data
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

class SigmoidPerceptron:
    def __init__(self, lr):
        self.w = None
        self.b = None
        self.lr = lr

    def forward(self, X):
        """Forward pass ONLY"""
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)  # Returns y_pred
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip for stability

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

# Initialize model
model = SigmoidPerceptron(learning_rate)

#Training
loss_history = []
n_samples, n_features = X.shape
model.w = np.random.randn(n_features) * 0.01
model.b = 0

def cross_entropy_loss(y_true, y_pred):
    """Binary cross-entropy loss"""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

for epoch in range(epochs+1):
    # Forward pass
    y_pred = model.forward(X)
    
    # Compute loss
    loss = cross_entropy_loss(y, y_pred)
    loss_history.append(loss)
    if epoch % 400 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # Gradients
    dw = model.lr * np.dot(X.T, (y_pred - y)) / n_samples
    db = model.lr * np.mean(y_pred - y)
    
    # Update
    model.w -= dw
    model.b -= db

# Test
p1=[0.15, 0.32]
p2=[0.65, 0.15]
pred1 = model.predict(p1)
pred2 = model.predict(p2)
print(f"{pred1:.4f} -> {pred1.round()}") # Expected: 0 (ham)
print(f"{pred2:.4f} -> {pred2.round()}") # Expected: 1 (spam)

# Plot
plot_loss(epochs, loss_history)
plot_training(X, y, model.w, model.b, p1, p2)