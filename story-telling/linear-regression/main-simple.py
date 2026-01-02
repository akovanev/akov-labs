import numpy as np 
import random
from plot import plot_simple  

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