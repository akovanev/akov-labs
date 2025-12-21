import numpy as np
from plot import plot_loss  
from plot import plot_training  

epochs = 2000 # Number of training epochs
learning_rate = 1e-2 # Learning rate for weight updates

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip for stability

class SigmoidPerceptron:
    def __init__(self, lr):
        self.w = None
        self.b = None
        self.lr = lr
        self.loss_history = []  # Track loss over epochs
    
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return sigmoid(z)
    
    def cross_entropy_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y, epochs):
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0
  
        for epoch in range(epochs+1):
            # Forward pass
            z = np.dot(X, self.w) + self.b
            y_pred = sigmoid(z)
            
            # Compute loss
            loss = self.cross_entropy_loss(y, y_pred)
            self.loss_history.append(loss)
            if epoch % 400 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

            # Gradients
            dw = self.lr * np.dot(X.T, (y_pred - y)) / n_samples
            db = self.lr * np.mean(y_pred - y)
            
            # Update
            self.w -= dw
            self.b -= db

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
sp = SigmoidPerceptron(learning_rate)
sp.fit(X, y, epochs)
p1=[0.15, 0.32]
p2=[0.65, 0.15]
pred1 = sp.predict(p1)
pred2 = sp.predict(p2)
print(f"{pred1:.4f} -> {pred1.round()}") # Expected: 0 (ham)
print(f"{pred2:.4f} -> {pred2.round()}") # Expected: 1 (spam)

plot_loss(epochs, sp.loss_history)
plot_training(X, y, sp.w, sp.b, p1, p2)