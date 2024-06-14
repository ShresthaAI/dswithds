import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw_bullseye(num_circles=7, outer_radius=1.0, hits=None, precision=None, threshold=0.1):
    fig, ax = plt.subplots(figsize=(4, 4))
    
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
    
    # Annotation for precision
    if precision is not None:
        ax.text(-outer_radius, -outer_radius - 0.15, f'Precision: {precision:.2f}', fontsize=12, ha='left')
    
    # Display legend
    ax.legend()

    return fig

@st.cache_data
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

@st.cache_data
def generate_hits(num_hits, mean_x, mean_y, std_dev_x, std_dev_y):
    if num_hits > 0:
        hits_x = np.round(np.random.normal(mean_x, std_dev_x, num_hits),2)
        hits_y = np.round(np.random.normal(mean_y, std_dev_y, num_hits),2)
        hits = list(zip(hits_x, hits_y))
    else:
        hits = []
    return hits

def bullseye_config(bullseye_num, outer_radius):
    num_hits = st.number_input(f"Number of Hit Points {bullseye_num}", 0, 100, 10, key=f"num_hits_{bullseye_num}")
    mean_x = st.slider(f"Mean X {bullseye_num}", -outer_radius, outer_radius, 0.0, key=f"mean_x_{bullseye_num}")
    mean_y = st.slider(f"Mean Y {bullseye_num}", -outer_radius, outer_radius, 0.0, key=f"mean_y_{bullseye_num}")
    std_dev_x = st.slider(f"Standard Deviation X {bullseye_num}", 0.0, outer_radius, 0.1, key=f"std_dev_x_{bullseye_num}")
    std_dev_y = st.slider(f"Standard Deviation Y {bullseye_num}", 0.0, outer_radius, 0.1, key=f"std_dev_y_{bullseye_num}")
    
    return num_hits, mean_x, mean_y, std_dev_x, std_dev_y

def generate_truth_table(predicted_hits, outer_radius, threshold=0.1):
    center = np.array([0, 0])
    distances = [np.linalg.norm(np.array(hit) - center) for hit in predicted_hits]
    is_in_target = [dist < threshold for dist in distances]
    truth_table_data = {
        'Hit Coordinates': predicted_hits,
        'Distance from Center': distances,
        'Within Target Area': is_in_target
    }
    return pd.DataFrame(truth_table_data)

def app():
    st.title("Bullseye Generator")
    
    # Fixed Bullseye configuration
    num_circles = 7
    outer_radius = 1.0
    num_bullseyes = 3

    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    bullseye_data = []

    for i, col in enumerate(columns):
        with col:
            st.header(f"Bullseye {i+1} Configuration")
            num_hits, mean_x, mean_y, std_dev_x, std_dev_y = bullseye_config(i+1, outer_radius)
            
            # Generate hit points
            hits = generate_hits(num_hits, mean_x, mean_y, std_dev_x, std_dev_y)

            # Compute precision and recall
            precision, recall = compute_precision_recall(hits)
            bullseye_data.append((hits, precision, recall))

            st.subheader(f"Bullseye {i+1}")
            fig = draw_bullseye(num_circles, outer_radius, hits, precision, threshold=outer_radius / num_circles)
            st.pyplot(fig)
            st.write(f"Recall: {recall:.2f}")
            
            # Generate and display truth table
            if hits:
                truth_table = generate_truth_table(hits, outer_radius, threshold=outer_radius / num_circles)
                st.subheader(f"Truth Table for Bullseye {i+1}")
                st.dataframe(truth_table)
            else:
                st.write("No hits generated.")

            # Display formula with values used for precision and recall calculation in LaTeX
            st.subheader("Precision and Recall Calculation")
            st.latex(r'''
                \text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}
            ''')
            st.latex(r'''
                \text{Recall} = \frac{\text{True Positives}}{\text{Total Hits}}
            ''')
            st.write("Calculating for Bullseye:")
            st.write(f"True Positives: {int(precision * len(hits))} (hits within target)")
            st.write(f"False Positives: {int((1 - precision) * len(hits))} (hits outside target)")
            st.write(f"Total Hits: {len(hits)} (all hits generated)")
            st.write(f"Threshold: {outer_radius / num_circles} (inner radius / number of bullseyes)")

    # Calculate combined recall
    total_true_positives = sum(data[1] * len(data[0]) for data in bullseye_data)
    total_hits = sum(len(data[0]) for data in bullseye_data)
    combined_recall = total_true_positives / total_hits if total_hits > 0 else 0

    st.subheader("Combined Recall")
    st.write(f"Total True Positives: {int(total_true_positives)}")
    st.write(f"Total Hits: {total_hits}")
    st.write(f"Combined Recall of all three Bullseyes: {combined_recall:.2f}")

if __name__ == "__main__":
    app()
