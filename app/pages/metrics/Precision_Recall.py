import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def draw_bullseye(num_circles=7, outer_radius=1.0, hits=None, precision=None, threshold=0.1):
    fig = go.Figure()
    
    # Plot each circle of the bullseye
    shapes =[]
    for i in range(num_circles):
        radius = outer_radius * (num_circles - i) / num_circles
        if i == num_circles - 1:
            color = 'yellow'  # Innermost circle (target area)
            label = 'Target'
        else:
            color = 'red' if i % 2 == 0 else 'white'
            label = None
        
        # Define the shape of the circle
        circle = dict(
            type="circle",
            xref="x",
            yref="y",
            fillcolor=color,
            line=dict(color='black', width=4),
            x0=-radius, y0=-radius,
            x1=radius, y1=radius,
            opacity=0.7,
            layer="below",
        )
        shapes.append(circle)
        # Add circle shape to the layout
    fig.update_layout(shapes=shapes)
        
    # Plot the hits
    if hits:
        hit_x, hit_y = zip(*hits)
        colors = ['blue' if np.linalg.norm(np.array(hit) - np.array([0, 0])) < threshold else 'black' for hit in hits]
        fig.add_trace(go.Scatter(
            x=hit_x, y=hit_y,
            mode='markers',
            marker=dict(color=colors, size=10, line=dict(color='black', width=1)),
            name='Hits',
            text=[f"x: {x:.2f}, y: {y:.2f}" for x, y in zip(hit_x, hit_y)],
            hoverinfo='text',
        ))
    
    # Set layout options
    fig.update_layout(
        xaxis=dict(range=[-outer_radius, outer_radius], visible=False),
        yaxis=dict(range=[-outer_radius, outer_radius], visible=False),
        plot_bgcolor='white',
        showlegend=True,
        title=dict(text='Bullseye', x=0.5, y=0.95, xanchor='center', yanchor='top'),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    # Annotation for precision
    if precision is not None:
        fig.add_annotation(
            x=-outer_radius, y=-outer_radius - 0.15,
            text=f'Precision: {precision:.2f}',
            showarrow=False,
            font=dict(size=12),
        )

    return fig

@st.cache_data
def compute_precision_recall_accuracy(predicted_hits, threshold=0.1):
    center = np.array([0, 0])
    distances = [np.linalg.norm(np.array(hit) - center) for hit in predicted_hits]
    
    true_positives = sum([dist <= threshold for dist in distances])  # Correct hits within the threshold
    false_positives = sum([dist > threshold for dist in distances])  # Incorrect hits outside the threshold
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / len(predicted_hits) if len(predicted_hits) > 0 else 0
    accuracy = true_positives / len(predicted_hits) if len(predicted_hits) > 0 else 0
    
    return precision, recall, accuracy

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
    is_in_target = [dist <= threshold for dist in distances]  # Adjusted to include boundary
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
            
            precision, recall = compute_precision_recall(hits,threshold=outer_radius / num_circles)
            bullseye_data.append((hits, precision, recall))

            st.subheader(f"Bullseye {i+1}")
            fig = draw_bullseye(num_circles, outer_radius, hits, precision, threshold=outer_radius / num_circles)
            st.plotly_chart(fig)

            st.write(f"Precision: {precision:.2f}")
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
