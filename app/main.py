# import streamlit as st
# from utils.utils import load_page
# st.set_page_config(page_title="DS with DS", layout="wide")



# st.sidebar.title("Navigation")
# pages = {
#     "Precision & Recall": "metrics/1_Precision_Recall",
#     "Vectors": "linalg/Vectors",
#     "About": "About"
# }

# choice = st.sidebar.selectbox("Go to", list(pages.keys()))

# # Assuming load_page takes page name as input
# load_page(pages[choice])


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def draw_bullseye(num_circles=7, outer_radius=1.0, hits=None, precision=None, recall=None, threshold=0.1):
    fig, ax = plt.subplots(figsize=(4,2))
    fig.set_figheight(3)
    
    for i in range(num_circles):
        radius = outer_radius * (num_circles - i) / num_circles
        if i == num_circles - 1:
            color = 'yellow'  # Innermost circle (target area)
            label = 'Target'
        else:
            color = 'red' if i % 2 == 0 else 'white'
            label = None
        circle = plt.Circle((0, 0), radius, color=color, ec='black', label=label)
        ax.add_artist(circle)
        
    if hits:
        hit_x, hit_y = zip(*hits)
        ax.plot(hit_x, hit_y, 'bo', label='Hits')
        
    ax.set_aspect('equal')
    ax.set_xlim(-outer_radius, outer_radius)
    ax.set_ylim(-outer_radius, outer_radius)
    ax.axis('off')
    
    # Annotation for precision and recall
    if precision is not None and recall is not None:
        ax.text(-outer_radius, -outer_radius - 0.15, f'Precision: {precision:.2f}, Recall: {recall:.2f}', fontsize=12, ha='left')
    
    
    # Display legend
    ax.legend()

    return fig


def compute_precision_recall(predicted_hits, threshold=0.1):
    true_positives = 0
    false_positives = 0
    
    # Center of the bullseye
    center = np.array([0, 0])
    
    for pred_hit in predicted_hits:
        distance = np.linalg.norm(np.array(pred_hit) - center)
        if distance < threshold:
            true_positives += 1
        else:
            false_positives += 1
            
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / len(predicted_hits) if len(predicted_hits) > 0 else 0
    
    return precision, recall

def app():
    st.title("Bullseye Generator")
    
    # Fixed Bullseye configuration
    num_circles = 7
    outer_radius = 1.0
    
    # Layout setup
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Configuration")
        num_hits = st.number_input("Number of Hit Points", 0, 100, 10)
        mean_x = st.slider("Mean X", -outer_radius, outer_radius, 0.0)
        mean_y = st.slider("Mean Y", -outer_radius, outer_radius, 0.0)
        std_dev_x = st.slider("Standard Deviation X", 0.0, outer_radius, 0.1)
        std_dev_y = st.slider("Standard Deviation Y", 0.0, outer_radius, 0.1)
        
    # Generate hit points
    if num_hits > 0:
        hits_x = np.random.normal(mean_x, std_dev_x, num_hits)
        hits_y = np.random.normal(mean_y, std_dev_y, num_hits)
        hits = list(zip(hits_x, hits_y))
    else:
        hits = []
    
    # Compute precision and recall
    precision, recall = compute_precision_recall(hits)
    
    with col2:
        # Draw the bullseye
        fig = draw_bullseye(num_circles, outer_radius, hits, precision, recall, threshold=outer_radius / num_circles)
        st.pyplot(fig)

if __name__ == "__main__":
    app()


