import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

embeddings = {
    'jazz': torch.tensor([0.2, 0.9, 0.1]),
    'classical': torch.tensor([0.3, 0.8, 0.25]),
    'pop': torch.tensor([0.1, 0.85, 0.15]),
    'granite': torch.tensor([0.9, 0.2, 0.3]),
    'crystal': torch.tensor([0.85, 0.25, 0.25]),
    'quartz': torch.tensor([0.8, 0.15, 0.35]),
    'cliff': torch.tensor([0.55, 0.18, 0.38]),
    'mountain': torch.tensor([0.42, 0.22, 0.36]),
    'rock': torch.tensor([0.5, 0.55, 0.25]),
}

# Rock variants placed directionally toward their semantic regions
embeddings.update({
    'rock_music': embeddings['rock'] + torch.tensor([0.0, 0.25, -0.05]),
    'rock_stone': embeddings['rock'] + torch.tensor([0.25, -0.25, 0.0]),
    'rock_geo': embeddings['rock'] + torch.tensor([-0.05, -0.3, 0.05]),
})

words = list(embeddings.keys())
vectors = torch.stack([embeddings[w] for w in words]).numpy()
rock_idx = words.index('rock')
rock_vec = vectors[rock_idx]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define clean polygon vertex order
music_indices = [words.index(w) for w in ['rock_music', 'classical', 'pop', 'jazz']]
stone_indices = [words.index(w) for w in ['rock_stone', 'quartz', 'granite', 'crystal']]
geo_indices   = [words.index(w) for w in ['rock_geo', 'cliff', 'mountain']]

# --- Draw filled areas ---
def draw_filled_region(indices, color):
    group_vectors = [vectors[i] for i in indices]
    poly = Poly3DCollection([group_vectors], alpha=0.25, facecolor=color, edgecolor=color, linewidths=1.5)
    ax.add_collection3d(poly)

draw_filled_region(music_indices, 'blue')
draw_filled_region(stone_indices, 'red')   # Fixed vertex ordering for proper parallelogram
draw_filled_region(geo_indices, 'orange')

# Scatter points
for i, word in enumerate(words):
    if word == 'rock':
        color, marker = 'lightgray', 'D'
    elif word.startswith('rock_music'):
        color, marker = 'lightblue', 'o'
    elif word.startswith('rock_stone'):
        color, marker = 'lightcoral', 's'
    elif word.startswith('rock_geo'):
        color, marker = 'orange', '^'
    elif word in ['jazz', 'classical', 'pop']:
        color, marker = 'lightblue', 'o'
    elif word in ['granite', 'crystal', 'quartz']:
        color, marker = 'lightcoral', 's'
    elif word in ['cliff', 'mountain']:
        color, marker = 'orange', '^'
    else:
        color, marker = 'gray', 'x'
    
    ax.scatter(vectors[i, 0], vectors[i, 1], vectors[i, 2],
               c=color, s=250, marker=marker, edgecolor='black', linewidth=1.5, alpha=1.0, zorder=10)
    label = 'rock' if word in ['rock_music', 'rock_stone', 'rock_geo'] else word
    ax.text(vectors[i, 0] + 0.02, vectors[i, 1] + 0.02, vectors[i, 2] + 0.02,
        label, fontsize=10, fontweight='bold')

# Connect only main rock with the variant rocks
variant_names = ['rock_music', 'rock_stone', 'rock_geo']
variant_colors = ['blue', 'red', 'orange']
for name, color in zip(variant_names, variant_colors):
    variant_vec = embeddings[name].numpy()
    ax.plot([rock_vec[0], variant_vec[0]],
            [rock_vec[1], variant_vec[1]],
            [rock_vec[2], variant_vec[2]],
            color=color, linestyle=':', linewidth=2, alpha=0.9)

# Aesthetics
ax.set_title('Embedding Vectors and the Semantic Contexts')
ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')

ax.view_init(elev=20, azim=50)
plt.tight_layout()
plt.show()
