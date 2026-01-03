import matplotlib.pyplot as plt

def plot_attention_trajectories(vocab, rock_orig, rock_q, rock_m, rock_c):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. DEFINE SPECIFIC STYLES
    # quartz: red square ('s'), cliff: orange triangle ('^')
    styles = {
        'quartz': {'color': 'red', 'marker': 's'},
        'cliff':  {'color': 'orange', 'marker': '^'},
        'jazz':  {'color': 'blue', 'marker': 'o'} # keeping music as blue circle
    }

    # 2. PLOT ORIGINAL ROCK (The Star)
    ax.scatter(*rock_orig, c='black', s=250, marker='*', label='rock', zorder=10)
    ax.text(*rock_orig, '  rock', size=12, fontweight='normal')

    # Mapping names to the numerical results
    results = {'quartz': rock_q, 'jazz': rock_m, 'cliff': rock_c}

    for name, style in styles.items():
        anchor = vocab[name].numpy()
        res = results[name].astype(float) # Ensure numerical for math
        color = style['color']
        marker = style['marker']
        
        # Plot the Anchor Word
        ax.scatter(*anchor, c=color, s=150, marker=marker, label=f'Anchor: {name}')
        ax.text(*anchor, f'  {name}', size=11)
        
        # Plot the Contextualized Rock (The result of attention)
        ax.scatter(*res, c=color, s=120, marker=marker, edgecolors='black')
        label = '     rock' if name == 'quartz' else '  rock'
        ax.text(*res, label, size=10)
        
        # Draw trajectory arrow from Original Rock to New Rock
        ax.quiver(rock_orig[0], rock_orig[1], rock_orig[2], 
                  res[0] - rock_orig[0], 
                  res[1] - rock_orig[1], 
                  res[2] - rock_orig[2], 
                  color=color, arrow_length_ratio=0.0, linewidth=2, linestyle='--')

    # 3. FINAL ADJUSTMENTS
    ax.set_xlabel('Geology (X)')
    ax.set_ylabel('Music (Y)')
    ax.set_zlabel('Elevation (Z)')
    ax.set_title('Attention Trajectories: How Context Shifts "Rock"')
    
    # Set axis limits to see the "shrinkage" clearly
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_zlim(0, 1.1)
    
    ax.view_init(elev=20, azim=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()