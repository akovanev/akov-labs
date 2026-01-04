import numpy as np
import matplotlib.pyplot as plt

def cross_entropy_loss(y_true, y_pred):
    # Small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Generate a range of predicted probabilities from 0 to 1
y_hat = np.linspace(0, 1, 500)

# Calculate loss for ground truth 1 and 0
loss_y1 = cross_entropy_loss(1, y_hat)
loss_y0 = cross_entropy_loss(0, y_hat)

# Plotting with swapped axes
plt.plot(loss_y1, y_hat, label='True Label = 1', color='blue', lw=2)
plt.plot(loss_y0, y_hat, label='True Label = 0', color='red', lw=2)

# Shading the region for True Label = 1
# Shading for True Label = 1
plt.fill_betweenx(y_hat, loss_y1, alpha=0.1, color='blue', 
                 label='Penalty Zone (Actual=1)')

# Shading for True Label = 0
plt.fill_betweenx(y_hat, loss_y0, alpha=0.1, color='red', 
                 label='Penalty Zone (Actual=0)')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

plt.title('Cross-Entropy Loss: Probability vs. Penalty', fontsize=14)
plt.xlabel('Loss Value', fontsize=12)
plt.ylabel(r'Predicted Probability ($\hat{y}$)', fontsize=12)
plt.xlim(0, 5)  # Limit X axis for visibility
plt.legend()
plt.grid(True, alpha=0.3)

# Highlight high loss area
plt.annotate('High Loss\n(Confident & Wrong)', xy=(3, 0.05), xytext=(3.5, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()