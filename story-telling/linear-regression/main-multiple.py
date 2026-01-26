import numpy as np
from plot import plot_multiple

def prepend_ones_column(X):
    """ Add a column of ones to X to account for the intercept term. """
    n_samples = X.shape[0]
    ones = np.ones((n_samples, 1))
    return np.append(ones, X, axis=1)

def fit_multiple_regression(X, y):
    """    
    X: Independent variables (n_samples, n_features)
    y: Dependent variable (n_samples, 1)
    """
    # 1. Prepare the matrix with intercept
    X = prepend_ones_column(X)
    
    # 2. Calculate the Normal Equation: (X^T * X)^-1 * X^T * y
    # X.T is the transpose
    # np.linalg.inv calculates the matrix inverse
    # @ is the matrix multiplication operator in Python
    xtx = X.T @ X
    xtx_inv = np.linalg.inv(xtx)
    xt_y = X.T @ y
    
    beta_hat = xtx_inv @ xt_y
    
    return beta_hat

# --- Example Usage ---
# Generate synthetic data (24 values)
np.random.seed(42)
square_meters = np.random.randint(100, 300, 24)
bedrooms = np.random.randint(2, 6, 24)

# True relationship: price = 10 + 0.85*square_meters + 15*Bedrooms + Noise
noise = np.random.normal(0, 20, 24)
prices = 10 + (0.85 * square_meters) + (15 * bedrooms) + noise

X_data = np.column_stack((square_meters, bedrooms))
y_data = prices

# Get the best estimate coefficients
beta_hat = fit_multiple_regression(X_data, y_data)

print(f"Intercept (Beta_0): {beta_hat[0]:.4f}")
print(f"Slope for square_meters (Beta_1): {beta_hat[1]:.4f}")
print(f"Slope for Bedrooms (Beta_2): {beta_hat[2]:.4f}")

# Make predictions
square_meters_col = X_data[:, 0]
bedrooms_col = X_data[:, 1]
X = prepend_ones_column(X_data)
y_pred = X @ beta_hat

plot_multiple(square_meters_col, bedrooms_col, y_data, y_pred, beta_hat)