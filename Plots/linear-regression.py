import numpy as np 
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_simple(x, y, y_pred, slope, intercept):
        plt.figure(figsize=(12, 8))
        plt.scatter(x, y, color='navy', label='$y$', s=90, alpha=0.85, edgecolors='royalblue', linewidth=1.2, zorder=2)

        # Blue predicted points
        plt.scatter(x, y_pred, color='lightskyblue', s=60, alpha=0.75, edgecolors='deepskyblue', linewidth=1, zorder=3, label='$\hat{{y}}$')
        plt.plot(x, y_pred, color='dodgerblue', linewidth=4, label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')

    
        plt.xlabel('X', fontsize=13, fontweight='bold')
        plt.ylabel('Y', fontsize=13, fontweight='bold')
        plt.title('Linear Regression: Actual $y$ vs Predicted $\hat{{y}}$', 
                fontsize=16, fontweight='bold', pad=25)
        plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

        # Blue-themed grid
        plt.grid(True, alpha=0.35, linestyle='--', linewidth=1, color='lightblue')
        plt.tight_layout()
        plt.show()

def mean(data):
    return sum(data) / len(data)

def covariance(x, y):
    mean_x, mean_y = mean(x), mean(y)
    return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / len(x)

def variance(x):
    mean_x = mean(x)
    return sum((xi - mean_x) ** 2 for xi in x) / len(x)

def linear_regression(x, y):
    slope = covariance(x, y) / variance(x)
    intercept = mean(y) - slope * mean(x)
    return intercept, slope

def predict(x, intercept, slope):
    return [intercept + slope * xi for xi in x]

# Usage
x = np.linspace(0, 20, 50)
y = [2.3 + 0.85 * i + random.gauss(0, 2.5) for i in x] 
intercept, slope = linear_regression(x, y)
print(f"Intercept: {intercept:.3f}, Slope: {slope:.3f}")
predictions = predict(x, intercept, slope)
print("Predictions:", [f"{pred:.3f}" for pred in predictions])

plot_simple(x, y, y_pred=predictions, slope=slope, intercept=intercept)