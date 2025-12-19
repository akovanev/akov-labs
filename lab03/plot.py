import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_simple(x, y, y_pred, slope, intercept):
        plt.figure(figsize=(12, 8))
        plt.scatter(x, y, color='navy', label='Actual data', s=90, alpha=0.85, 
                edgecolors='royalblue', linewidth=1.2, zorder=2)
        plt.plot(x, y_pred, color='dodgerblue', linewidth=4, label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')

        # Blue predicted points
        plt.scatter(x, y_pred, color='lightskyblue', s=60, alpha=0.75, 
                edgecolors='deepskyblue', linewidth=1, zorder=3, label='Predictions')

        # Blue confidence band
        y_err = [1.5] * len(x)
        plt.fill_between(x, [p-1.5 for p in y_pred], [p+1.5 for p in y_pred], 
                        color='lightblue', alpha=0.4, label='Confidence Interval')

        plt.xlabel('X Values', fontsize=13, fontweight='bold')
        plt.ylabel('Y Values', fontsize=13, fontweight='bold')
        plt.title('Linear Regression: Actual vs Predicted', 
                fontsize=16, fontweight='bold', pad=25)
        plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

        # Blue-themed grid
        plt.grid(True, alpha=0.35, linestyle='--', linewidth=1, color='lightblue')
        plt.tight_layout()
        plt.show()

def plot_multiple(sqft, bedrooms, y_data, y_pred, beta):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot Actual Data (Red Dots)
    actual = ax.scatter(sqft, bedrooms, y_data, color='red', label='Actual Data', s=50)

    # 2. Plot Predicted Data (Blue Crosses)
    predicted = ax.scatter(sqft, bedrooms, y_pred, color='blue', marker='x', label='Predicted Data', s=50)

    # 3. Draw lines (Residuals)
    for i in range(len(y_data)):
        ax.plot([sqft[i], sqft[i]], [bedrooms[i], bedrooms[i]], 
                [y_data[i], y_pred[i]], color='gray', linestyle='--', alpha=0.5)

    # 4. Create the Regression Surface (The Plane)
    x_surf = np.linspace(sqft.min(), sqft.max(), 20)
    y_surf = np.linspace(bedrooms.min(), bedrooms.max(), 20)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = beta[0] + beta[1] * x_surf + beta[2] * y_surf
    
    # We use a variable 'plane' so we can reference it in the legend
    plane = ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.3, color='cyan', label='Regression Plane')
    
    # Fix for plot_surface legend: Create a proxy for the legend
    # Because plot_surface handles labels differently
    plane_proxy = mpatches.Patch(color='cyan', alpha=0.3, label='Regression Plane')

    # Labels
    ax.set_xlabel('SqFt')
    ax.set_ylabel('Bedrooms')
    ax.set_zlabel('Price ($k)')
    ax.set_title('Multiple Linear Regression')
    
    # 5. Add Legend
    # We pass the proxies/handles to the legend function
    ax.legend(handles=[actual, predicted, plane_proxy])
    
    plt.show()