import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return 2*x**2 + 3*y**2

def grad_f(x, y):
    return 4*x, 6*y

def grad_descent(x0, y0, eta, tolerance, max_steps):
    x, y = x0, y0
    step = 0
    cost = f(x, y)
    trajectory = [(x, y, cost)] # to store the trajectory for plotting
    while cost > tolerance and step < max_steps:
        grad_x, grad_y = grad_f(x, y)
        x -= eta * grad_x
        y -= eta * grad_y
        cost = f(x, y)
        trajectory.append((x, y, cost))
        step += 1
        if step % 5 == 0 or cost <= tolerance:
            print(f"Step {step}: ({x:.3f}, {y:.3f}), cost={cost:.3f}")
    return trajectory

def plot(trajectory):
    # Create contour plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Surface
    x_surf = np.linspace(-3, 3, 50)
    y_surf = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_surf, y_surf)
    Z = 2*X**2 + 3*Y**2
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    # Trajectory points and path
    traj_x = [t[0] for t in trajectory]
    traj_y = [t[1] for t in trajectory]
    traj_z = [t[2] for t in trajectory]

    # Plot full trajectory line
    ax.plot(traj_x, traj_y, traj_z, 'r-', linewidth=3, label='Descent Path', alpha=0.8)

    # Scatter all points (smaller for intermediates)
    ax.scatter(traj_x, traj_y, traj_z, c='red', s=30, alpha=0.6)

    # Highlight start and end
    ax.scatter(traj_x[0], traj_y[0], traj_z[0], color='green', s=150, label='Start (2,-3,35)')
    ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], color='gold', s=200, marker='*', label=f'End ({traj_x[-1]:.2f},{traj_y[-1]:.2f})')

    # Gradient at start (for reference)
    scale = 0.3
    ax.quiver(2, -3, 35, 8*scale, -18*scale, 0, color='brown', arrow_length_ratio=0.1, 
            label='Initial ∇=(8,-18)', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Gradient Descent Trajectory: z=2x²+3y²\nη=0.05, starts at (2,-3,35)')
    ax.legend()
    plt.show()
   
        
trajectory = grad_descent(2.0, -3.0, eta=0.05, tolerance=1e-3, max_steps=100)
plot(trajectory)
