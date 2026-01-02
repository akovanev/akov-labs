import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

def draw_improved_neural_net():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Positions: {Layer Name: [(x, y), ...]}
    pos = {
        'Input': [(2, 7.5), (2, 5.5), (2, 3.5)],
        'Hidden': [(6, 8.5), (6, 6.5), (6, 4.5), (6, 2.5)],
        'Output': [(10, 5.5)],
        'Target': [(10, 8.5)], # Ground Truth y
        'Loss': [(12.5, 7.0)]  # The Objective Function
    }

    # 1. Draw Connections (Arrows)
    # Forward Pass
    for start in pos['Input']:
        for end in pos['Hidden']:
            ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='#2b5a91', lw=1, alpha=0.3))
    
    for start in pos['Hidden']:
        ax.annotate('', xy=pos['Output'][0], xytext=start, arrowprops=dict(arrowstyle='->', color='#2b5a91', lw=1.5, alpha=0.5))

    # To Loss
    ax.annotate('', xy=pos['Loss'][0], xytext=pos['Output'][0], arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=pos['Loss'][0], xytext=pos['Target'][0], arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Backward Pass (Gradients)
    # From Loss to Output
    ax.annotate('', xy=pos['Output'][0], xytext=pos['Loss'][0], 
                arrowprops=dict(arrowstyle='->', color='#d62728', ls='--', lw=2, connectionstyle="arc3,rad=0.2"))
    
    # Through Hidden Layers
    for h_pos in pos['Hidden']:
        ax.annotate('', xy=h_pos, xytext=pos['Output'][0], 
                    arrowprops=dict(arrowstyle='->', color='#d62728', ls='--', lw=1.5, alpha=0.6, connectionstyle="arc3,rad=0.1"))

    # 2. Draw Nodes
    def add_node(coords, label, color, ec='black', shape='circle'):
        if shape == 'circle':
            circle = Circle(coords, 0.4, fc=color, ec=ec, zorder=5)
            ax.add_patch(circle)
        else: # Loss Box
            box = FancyBboxPatch((coords[0]-0.6, coords[1]-0.5), 1.2, 1.0, 
                                 boxstyle="round,pad=0.1", fc=color, ec=ec, zorder=5)
            ax.add_patch(box)
        ax.text(coords[0], coords[1], label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=6)

    # Draw Input, Hidden, Output
    for i, p in enumerate(pos['Input']): add_node(p, rf'$x_{i+1}$', '#a1c9f4')
    for i, p in enumerate(pos['Hidden']): add_node(p, rf'$h_{i+1}$', '#8de5a1')
    add_node(pos['Output'][0], r'$\hat{y}$', '#ff9f9b')
    
    # Draw Target and Loss
    add_node(pos['Target'][0], r'$y$', '#f7f7f7', ec='gray')
    add_node(pos['Loss'][0], 'Loss\n$L(y, \hat{y})$', '#ffcc00', shape='box')

    # Annotations
    ax.text(2, 9.2, "Features", ha='center', fontsize=12, color='gray')
    ax.text(10, 9.2, "Labels", ha='center', fontsize=12, color='gray')
    ax.text(7, 0.5, "Backpropagation starts at the Loss and flows toward the inputs to update weights.", 
            ha='center', fontsize=11, style='italic', color='#444')

    plt.title("End-to-End Deep Learning Flow", fontsize=18, fontweight='bold', pad=30)
    plt.show()

draw_improved_neural_net()