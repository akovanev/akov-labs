import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

def draw_forward_pass():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Positions
    pos = {
        'Input': [(2, 7.5), (2, 5.5), (2, 3.5)],
        'Hidden': [(6, 8.5), (6, 6.5), (6, 4.5), (6, 2.5)],
        'Output': [(10, 5.5)],
        'Target': [(10, 8.5)],
        'Loss': [(12.5, 7.0)]
    }

    # 1. Draw Forward Connections ONLY
    # Input to Hidden
    for start in pos['Input']:
        for end in pos['Hidden']:
            ax.annotate('', xy=end, xytext=start, 
                        arrowprops=dict(arrowstyle='-|>', color='#2b5a91', lw=1.5, alpha=0.4))
    
    # Hidden to Output
    for start in pos['Hidden']:
        ax.annotate('', xy=pos['Output'][0], xytext=start, 
                    arrowprops=dict(arrowstyle='-|>', color='#2b5a91', lw=2, alpha=0.6))

    # Output/Target to Loss
    ax.annotate('', xy=pos['Loss'][0], xytext=pos['Output'][0], 
                arrowprops=dict(arrowstyle='-|>', color='black', lw=2, ls=':'))
    ax.annotate('', xy=pos['Loss'][0], xytext=pos['Target'][0], 
                arrowprops=dict(arrowstyle='-|>', color='black', lw=2, ls=':'))

    # 2. Add Mathematical Annotations
    ax.text(4, 9, r'$W^{(1)}, b^{(1)}$', fontsize=14, color='#2b5a91', ha='center')
    ax.text(8, 7.5, r'$W^{(2)}, b^{(2)}$', fontsize=14, color='#2b5a91', ha='center')

    # 3. Draw Nodes
    def add_node(coords, label, color, ec='black', shape='circle'):
        if shape == 'circle':
            circle = Circle(coords, 0.4, fc=color, ec=ec, zorder=5)
            ax.add_patch(circle)
        else:
            box = FancyBboxPatch((coords[0]-0.6, coords[1]-0.5), 1.2, 1.0, 
                                 boxstyle="round,pad=0.1", fc=color, ec=ec, zorder=5)
            ax.add_patch(box)
        ax.text(coords[0], coords[1], label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=6)

    for i, p in enumerate(pos['Input']): add_node(p, rf'$x_{i+1}$', '#a1c9f4')
    for i, p in enumerate(pos['Hidden']): add_node(p, rf'$h_{i+1}$', '#8de5a1')
    add_node(pos['Output'][0], r'$\hat{y}$', '#ff9f9b')
    add_node(pos['Target'][0], r'$y$', '#f7f7f7', ec='gray')
    add_node(pos['Loss'][0], 'Compute\nError', '#ffcc00', shape='box')

    # Legend & Title
    ax.text(7, 0.5, "Forward Pass: Feature Extraction $\longrightarrow$ Prediction $\longrightarrow$ Evaluation", 
            ha='center', fontsize=13, fontweight='bold', color='#2b5a91')
    
    plt.title("Neural Network: The Forward Pass", fontsize=18, fontweight='bold', pad=30)
    plt.show()

draw_forward_pass()