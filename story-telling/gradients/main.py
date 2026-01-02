from plot import plot

def f(x, y):
    return 2*x**2 + 3*y**2

def grad_f(x, y):
    return 4*x, 6*y

def grad_descent(x0, y0, alpha, tolerance, max_steps):
    x, y = x0, y0
    step = 0
    cost = f(x, y)
    trajectory = [(x, y, cost)] # store the trajectory for plotting
    while cost > tolerance and step < max_steps:
        grad_x, grad_y = grad_f(x, y)
        x -= alpha * grad_x
        y -= alpha * grad_y
        cost = f(x, y)
        trajectory.append((x, y, cost))
        step += 1
        if step % 5 == 0 or cost <= tolerance:
            print(f"Step {step}: ({x:.3f}, {y:.3f}), cost={cost:.3f}")
    return trajectory
   
        
trajectory = grad_descent(2.0, -3.0, alpha=0.05, tolerance=1e-3, max_steps=100)
plot(trajectory)
