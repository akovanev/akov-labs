import numpy as np
import matplotlib.pyplot as plt

def plot_airport_classification():
    np.random.seed(42)
    n_airports = 40
    
    # Generate Synthetic Data for 4 Airport Types
    # x1: Runway Length (ft), x2: Daily Flights
    
    # 1. Regional Hub (Circle): Short runways, low traffic
    regional = np.random.multivariate_normal([6000, 50], [[400**2, 0], [0, 15**2]], n_airports)
    
    # 2. Cargo Center (Square): Long runways (for heavy jets), low frequency
    cargo = np.random.multivariate_normal([13000, 80], [[800**2, 0], [0, 20**2]], n_airports)
    
    # 3. Metropolitan (Triangle): Long runways, extremely high traffic
    intl = np.random.multivariate_normal([11000, 1200], [[1000**2, 0], [0, 250**2]], n_airports)
    
    # 4. Executive/Private (Diamond): Medium runways, high frequency of small jets
    private = np.random.multivariate_normal([8000, 600], [[600**2, 0], [0, 100**2]], n_airports)

    # Setup the Plot
    plt.figure(figsize=(12, 8))
    
    # Mapping Categories to Markers and Colors
    categories = [
        (regional, 'o', '#a1c9f4', 'Regional Hub'),
        (cargo, 's', '#ffb482', 'Cargo Center'),
        (intl, '^', '#8de5a1', 'Metropolitan Intl'),
        (private, 'D', '#ff9f9b', 'Executive/Private')
    ]

    for data, marker, color, label in categories:
        plt.scatter(data[:, 0], data[:, 1], 
                    marker=marker, 
                    c=color, 
                    edgecolors='black', 
                    s=120, 
                    label=label, 
                    alpha=0.8,
                    zorder=3)

    # Labels and Scaling
    plt.title("Aviation Data: Airport Categorization by Capacity & Traffic", fontsize=16, fontweight='bold')
    plt.xlabel("Runway Length (Feet)", fontsize=12)
    plt.ylabel("Average Daily Flights", fontsize=12)
    
    # Setting limits to show the magnitude gap
    plt.xlim(4000, 16000)
    plt.ylim(0, 2000)
    
    plt.grid(True, linestyle='--', alpha=0.4, zorder=0)
    plt.legend(title="Airport Class", frameon=True, shadow=True, fontsize=11)
    
    # Annotating the 'Magnitude Gap'
    plt.annotate('Large Magnitude (Thousands)', xy=(10000, -100), xycoords='data', 
                 xytext=(0, -40), textcoords='offset points', ha='center', color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.annotate('Small Magnitude (Hundreds)', xy=(3500, 1000), xycoords='data', 
                 xytext=(-60, 0), textcoords='offset points', rotation=90, va='center', color='green',
                 arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    plt.show()

plot_airport_classification()