import matplotlib.pyplot as plt
import numpy as np

# XOR data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# Plot
plt.figure(figsize=(8, 6))
colors = ['red' if label == 0 else 'blue' for label in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2)

# Labels and annotations
for i, (x1, x2) in enumerate(X):
    plt.annotate(f'({x1},{x2}) → {y[i]}', (x1, x2), xytext=(5, 5), 
                textcoords='offset points', fontsize=12, ha='left')

plt.xlabel('x1', fontsize=14)
plt.ylabel('x2', fontsize=14)
plt.title('XOR Problem: Non-Linearly Separable', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.xticks([0, 1])
plt.yticks([0, 1])

# Attempted decision boundaries (show failure)
plt.plot([0.5, 0.5], [-0.1, 1.1], 'k--', alpha=0.5)
plt.plot([-0.1, 1.1], [0.5, 0.5], 'k--', alpha=0.5)

plt.tight_layout()
plt.show()
