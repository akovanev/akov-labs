import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate x values spanning several standard deviations
x = np.linspace(-4, 4, 1000)
# Compute PDF for standard normal distribution (mu=0, sigma=1)
y = norm.pdf(x, loc=0, scale=1)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', linewidth=2, label='N(0,1)')
plt.fill_between(x, y, alpha=0.3, color='blue')
plt.axvline(0, color='red', linestyle='--', label='μ = 0')
plt.axvline(1, color='green', linestyle='--', alpha=0.7, label='±1σ')
plt.axvline(-1, color='green', linestyle='--', alpha=0.7)

plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Standard Normal Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
