import matplotlib.pyplot as plt
import numpy as np

def draw_tensor(ax, shape, title, color, offset_multiplier=1.4):
    ax.view_init(elev=20, azim=35)
    
    # Rank 0 - 3
    if len(shape) <= 3:
        vis_shape = list(shape)
        while len(vis_shape) < 3: vis_shape.append(1)
        x, y, z = np.indices(np.array(vis_shape) + 1)
        facecolors = np.empty(tuple(vis_shape) + (4,))
        facecolors[:] = color
        ax.voxels(x, y, z, np.ones(vis_shape), facecolors=facecolors, edgecolors='black', linewidth=0.5)
        ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.set_zlim(0, 6)
        
    # Rank 4
    elif len(shape) == 4:
        num_blocks = shape[0]
        inner = shape[1:]
        for i in range(num_blocks):
            offset = i * (inner[0] * offset_multiplier)
            x, y, z = np.indices(np.array(inner) + 1)
            ax.voxels(x + offset, y, z, np.ones(inner), facecolors=color, edgecolors='black', linewidth=0.3)
        ax.set_xlim(0, num_blocks * inner[0] * offset_multiplier)
        
    # Rank 5: A grid of 3D volumes
    elif len(shape) == 5:
        rows = shape[0]
        cols = shape[1]
        inner = shape[2:]
        for r in range(rows):
            for c in range(cols):
                # Offset in both X and Y directions to create a grid
                off_x = c * (inner[0] * offset_multiplier)
                off_y = r * (inner[1] * offset_multiplier)
                x, y, z = np.indices(np.array(inner) + 1)
                ax.voxels(x + off_x, y + off_y, z, np.ones(inner), facecolors=color, edgecolors='black', linewidth=0.2)
        ax.set_xlim(0, cols * inner[0] * offset_multiplier)
        ax.set_ylim(0, rows * inner[1] * offset_multiplier)
        ax.set_zlim(0, 6)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

fig = plt.figure(figsize=(18, 10))

# ROW 1
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
draw_tensor(ax1, (1,), r'Scalar ($0^{th}$ Order)' + '\n' + r'$s \in \mathbb{R}$', [1, 0.4, 0.4, 0.7])

ax2 = fig.add_subplot(2, 3, 2, projection='3d')
draw_tensor(ax2, (5,), r'Vector ($1^{st}$ Order)' + '\n' + r'$v \in \mathbb{R}^{5}$', [0.4, 0.8, 0.4, 0.7])

ax3 = fig.add_subplot(2, 3, 3, projection='3d')
draw_tensor(ax3, (5, 4), r'Matrix ($2^{nd}$ Order)' + '\n' + r'$M \in \mathbb{R}^{5 \times 4}$', [0.4, 0.4, 1, 0.7])

# ROW 2
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
draw_tensor(ax4, (5, 4, 3), r'Tensor ($3^{rd}$ Order)' + '\n' + r'$\mathcal{T} \in \mathbb{R}^{5 \times 4 \times 3}$', [0.9, 0.7, 0.1, 0.5])

# ax5: Rank 4 (Line of Cubes)
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
draw_tensor(ax5, (3, 3, 3, 3), r'Rank 4 Tensor' + '\n' + r'$\mathcal{X} \in \mathbb{R}^{3 \times 3 \times 3 \times 3}$', [0.7, 0.3, 0.9, 0.4])

# ax6: Rank 5 (Grid of Cubes)
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
draw_tensor(ax6, (2, 2, 2, 2, 2), r'Rank 5 Tensor' + '\n' + r'$\mathcal{Y} \in \mathbb{R}^{2 \times 2 \times 2 \times 2 \times 2}$', [0.2, 0.8, 0.8, 0.3])

plt.tight_layout()
plt.show()