import numpy as np
from plot import plot 
import torch

def f(x, y):
    return 2*x**2 + 3*y**2

def grad_descent(x0, y0, eta, tolerance, max_steps):
    x = torch.tensor(x0, requires_grad=True)
    y = torch.tensor(y0, requires_grad=True)
    step = 0
    cost = f(x, y)
    trajectory = [(x.item(), y.item(), cost.item())]
    while cost > tolerance and step < max_steps:
        cost.backward()
        with torch.no_grad():
            x -= eta * x.grad
            y -= eta * y.grad
            x.grad.zero_()  # Clear gradients
            y.grad.zero_()
        cost = f(x, y)
        trajectory.append((x.item(), y.item(), cost.item()))
        step += 1
        if step % 5 == 0 or cost <= tolerance:
            print(f"Step {step}: ({x:.3f}, {y:.3f}), cost={cost:.3f}")
    return trajectory

trajectory = grad_descent(2.0, -3.0, eta=0.05, tolerance=1e-3, max_steps=100)
plot(trajectory)
