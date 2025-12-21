import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_loss(epochs, loss_history):
    plt.plot(range(len(loss_history)), loss_history) 
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss over Epochs")
    plt.grid()
    plt.show()

def plot_training(X, y, w, b, p1, p2):
    # scatter training data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    plt.xlabel("x1 (spam words fraction)")
    plt.ylabel("x2 (capital letters fraction)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # decision boundary: sigmoid(w1*x1 + w2*x2 + b) = 0.5 -> w1*x1 + w2*x2 + b = 0
    x1_min, x1_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
    x1_grid = np.linspace(x1_min, x1_max, 100)

    # avoid division by zero if w[1] ≈ 0
    if abs(w[1]) > 1e-8:
        x2_boundary = -(w[0] * x1_grid + b) / w[1]
        plt.plot(x1_grid, x2_boundary, "k--", label="decision boundary")

    # also plot your two test points
    plt.scatter(p1[0], p1[1], c="green", marker="x", s=80, label="test ham")
    plt.scatter(p2[0], p2[1], c="purple", marker="x", s=80, label="test spam")

    plt.legend()
    plt.title("Sigmoid perceptron decision boundary")
    plt.grid()
    plt.show()