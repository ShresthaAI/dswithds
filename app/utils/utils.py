import numpy as np
import matplotlib.pyplot as plt

def get_color():
    return((np.random.random(),np.random.random(),np.random.random()))

def visualize_vector(title="",new_basis=np.array([[1,0],[0,1]]), show=True,colors=['red','green','blue'],grid_length = 5, **kwargs):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Original basis
    original_basis = np.array([[1, 0], [0, 1]])
    axes = np.array([np.multiply(new_basis, -1*grid_length), np.multiply(new_basis, grid_length)])
    
    # Plot axes
    ax.plot(axes[:, 0][:, 0], axes[:, 0][:, 1], c='k', linewidth=1.5)
    ax.plot(axes[:, 1][:, 0], axes[:, 1][:, 1], c='k', linewidth=1.5)
    
    # Plot the grid
    for i in range(-1*(grid_length-1), grid_length):
        ax.plot(axes[:, 0][:, 0], axes[:, 0][:, 1] + i, c='grey', linewidth=0.5)
        ax.plot(axes[:, 1][:, 0] + i, axes[:, 1][:, 1], c='grey', linewidth=0.5)
    
    # Change basis
    if not np.all(original_basis == new_basis):
        transformation_matrix = np.linalg.inv(original_basis) @ new_basis
        for i, v in kwargs.items():
            kwargs[i] = transformation_matrix @ v
    
    # Plot all vectors
    for i,(k, v) in enumerate(kwargs.items()):
        color = np.random.rand(3)
        if i >= len(colors):
            color = np.random.rand(3)
        else:
            color = colors[i]
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=color, linewidth=2))
        ax.annotate(k, xy=v/2, xytext=[v[0]/2, (v[1]/2) + 0.8], c=color)
        ax.annotate(f"({v[0]}, {v[1]})", xy=v, xytext=v, c=color)
    
    # Set ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    ax.set_title(title)
    
    if show:
        return fig


def normalize_vector(v):
    assert type(v) == type(np.array([]))
    mag = np.sqrt(np.sum(v*v))
    norm = np.divide(v,mag)
    norm = np.round(norm,2)
    return norm
