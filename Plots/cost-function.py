import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random


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

def plot_simple(x, y, y_pred, slope, intercept):
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='navy', label='$y$', s=90, alpha=0.85, 
                edgecolors='royalblue', linewidth=1.2, zorder=2)
    
    # Predicted points (fixed edgecolors)
    plt.scatter(x, y_pred, color='lightskyblue', s=60, alpha=0.75, 
                edgecolors='deepskyblue', linewidth=1, zorder=3, label='$\hat{{y}}$')
    
    plt.plot(x, y_pred, color='dodgerblue', linewidth=4, label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')

    
    # Dashed residuals + diff labels
    for i in range(len(x)):
        diff = y[i] - y_pred[i]
        plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'k--', alpha=0.5, linewidth=1.5)
        mid_x = x[i]
        mid_y = (y[i] + y_pred[i]) / 2
        plt.annotate(f'{diff:.1f}', (mid_x, mid_y), xytext=(5, 0), 
                     textcoords='offset points', ha='left', fontsize=9, 
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7))
    
    # MSE Cost Function (fixed)
    # MSE Cost Function (fixed braces)
    mse = np.mean((np.array(y) - np.array(y_pred))**2)
    rmse = np.sqrt(mse)
    label = (
        rf'$J(\theta) = \frac{{1}}{{m}} \sum (y_i - \hat{{y}}_i)^2 = {mse:.3f}$' + '\n' +
        rf'$\sigma = \sqrt{{MSE}} = \sqrt{{{mse:.3f}}} = {rmse:.3f}$'
     )
    plt.text(0.93, 0.98, label, 
         transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9),
         ha='right', va='top')

    plt.xlabel('X', fontsize=13, fontweight='bold')
    plt.ylabel('Y', fontsize=13, fontweight='bold')
    plt.title('Actual $y$ vs Predicted $\hat{{y}}$ (MSE = J(θ), RMSE=σ)', fontsize=16, fontweight='bold', pad=25)
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.35, linestyle='--', linewidth=1, color='lightblue')
    plt.tight_layout()
    plt.show()

x = np.linspace(0, 20, 5)
y = [2.3 + 0.85 * i + random.gauss(0, 2.5) for i in x] 
intercept, slope = linear_regression(x, y)
print(f"Intercept: {intercept:.3f}, Slope: {slope:.3f}")
predictions = predict(x, intercept, slope)
print("Predictions:", [f"{pred:.3f}" for pred in predictions])

plot_simple(x, y, y_pred=predictions, slope=slope, intercept=intercept)