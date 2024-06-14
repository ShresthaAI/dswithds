import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def load_page(page_name):
    if page_name == "home":
        st.title("Welcome to the Multi-Page Streamlit App")
        st.write("Select a page from the sidebar to get started.")
    elif page_name == "metrics/1_Precision_Recall":
        from pages.metrics import Precision_Recall as page
        page.app()
        return True
    elif page_name == "linalg/Vectors":
        from pages.linalg import Vectors as page
        page.app()
        return True
    elif page_name == "about":
        from pages.about import about as page
        page.app()
        return True
    else:
        return False
    
def get_color():
    return((np.random.random(),np.random.random(),np.random.random()))

def normalize_vector(v):
    assert type(v) == type(np.array([]))
    mag = np.sqrt(np.sum(v*v))
    norm = np.divide(v,mag)
    norm = np.round(norm,2)
    return norm


def visualize_vector(title="", new_basis=np.array([[1, 0], [0, 1]]), show=False, show_original_basis=True, colors=['red', 'green', 'blue'], grid_size=5, **kwargs):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Original basis
    original_basis = np.array([[1, 0], [0, 1]])
    
    # Check if basis vectors are linearly independent
    if np.linalg.matrix_rank(new_basis) < 2:
        print("Error: Basis vectors are linearly dependent.")
        return None

    # Calculate axes for new basis and plot
    axes = np.array([np.multiply(new_basis, -1 * grid_size), np.multiply(new_basis, grid_size)])
    plt.plot(axes[:, 0][:, 0], axes[:, 0][:, 1], c='k', linewidth=1.5)
    plt.plot(axes[:, 1][:, 0], axes[:, 1][:, 1], c='k', linewidth=1.5)
    for i in range(-(grid_size - 1), grid_size):
        plt.plot(axes[:, 0][:, 0], axes[:, 0][:, 1] + i, c='grey', linewidth=0.5)
        plt.plot(axes[:, 1][:, 0] + i, axes[:, 1][:, 1], c='grey', linewidth=0.5)
    
    # Plot original basis and grid if requested
    if show_original_basis:
        original_axes = np.array([np.multiply(original_basis, -1 * grid_size), np.multiply(original_basis, grid_size)])
        plt.plot(original_axes[:, 0][:, 0], original_axes[:, 0][:, 1], c='grey', linestyle='--', linewidth=1.5)
        plt.plot(original_axes[:, 1][:, 0], original_axes[:, 1][:, 1], c='grey', linestyle='--', linewidth=1.5)
        for i in range(-(grid_size - 1), grid_size):
            plt.plot(original_axes[:, 0][:, 0], original_axes[:, 0][:, 1] + i, c='grey', linestyle='--', linewidth=0.5)
            plt.plot(original_axes[:, 1][:, 0] + i, original_axes[:, 1][:, 1], c='grey', linestyle='--', linewidth=0.5)

    # Change basis and plot vectors
    for i, v in kwargs.items():
        color = get_color()
        # Check if basis is the identity matrix, no transformation needed
        if np.all(new_basis == np.eye(2)):
            v_new_basis = v
        else:
            v_new_basis = np.linalg.inv(new_basis) @ v
        plt.annotate("", xy=v_new_basis, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=color, linewidth=2))
        plt.annotate(i, xy=v_new_basis / 2, xytext=[v_new_basis[0] / 2, (v_new_basis[1] / 2) + 0.8], c=color)
        plt.annotate(f"({v[0]},{v[1]})", xy=v_new_basis, xytext=v_new_basis, c=color)

        if not np.all(original_basis==new_basis):
            plt.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=color, linewidth=2, linestyle='--'))
            plt.annotate("", xy=original_basis[0], xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='green', linewidth=2, linestyle='--',alpha=0.5))
            plt.annotate("", xy=original_basis[1], xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='blue', linewidth=2, linestyle='--',alpha=0.5))
            plt.annotate("", xy=new_basis[0], xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='green', linewidth=2,alpha=0.5 ))
            plt.annotate("", xy=new_basis[1], xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='blue', linewidth=2,alpha=0.5 ))

    # Set limits based on grid size
    ax.set_xlim([-grid_size, grid_size])
    ax.set_ylim([-grid_size, grid_size])
    
    # Set ticks and title
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    
    if show:
        plt.show()
    else:
        return fig

