import os 
import sys 

import streamlit as st
import time
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, parent_dir)

from utils import utils


st.set_page_config(page_title="Linear Algebra", page_icon="")


st.markdown("# Linear Algebra")

st.sidebar.header("Linear Algebra")

st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

st.markdown("## Vectors")

st.markdown(
    """
$$
Cow = [10\space50\space20]
\\newline
Dog = [12\space8\space10]
\\newline
Cow-Dog = [10-12\space50-8\space20-10] = [-2\space42\space10]
$$
"""
)


st.markdown(
    """
    ### Ordered list of numbers

$$
N = \\begin{pmatrix}
n_1\\\\
n_2\\\\
\\end{pmatrix}
$$

where $n_1$ = firing rate of neruon 1 

$n_2$ = firing rate neuron 2
"""
)

st.code(
    '''
    n1 = 6
    n2 = 9
    N = np.array([n1, n2])
    N # output : np.array([6,9])
'''
)

st.markdown('''### An arrow''')

vector = [st.number_input("X1",value=1.0,min_value=-5.0,max_value=5.0,step=0.5,key='x1'),st.number_input("Y1",value=1.0,min_value=-5.0,max_value=5.0,step=0.5,key='y1')]
st.write(f"[{vector[0]} {vector[1]}]")
# Example usage
fig = utils.visualize_vector(title="Vector Visualization", show=False,
                       A=np.array(vector),colors=['red'])

st.pyplot(fig=fig)


st.markdown(
    """
    ## Normalizing Vector
Normalization of vector refers to transforming a vector to length of 1 unit while preserving its direction. We do this by dividing each component of the vector by its magnitude or length.

$$

\\hat{v} = \\frac{v}{||v||}
\\newline
$$
$$

||v|| = \sqrt{v_1^2+v_2^2+...+v_n^2}
$$

where $||v||$ is the magnitude of the vector. Normalization is often used in various computational and mathematical applications inclduing machine learning, signal processing and physics.
"""
)

vector = np.array([st.number_input("X1",value=1.0,min_value=-5.0,max_value=5.0,step=0.5,key='norm_x1'),st.number_input("y1",value=1.0,min_value=-5.0,max_value=5.0,step=0.5,key='norm_y1')])
st.write(f"**Normalized vector :**[{2*vector[0]} {2*vector[1]}]")
norm = utils.normalize_vector(vector)
fig = utils.visualize_vector(v=vector,norm=norm,title="Arrow representation of vector and normalized form ")
st.pyplot(fig=fig)

st.markdown("## Vector Operations")
st.markdown("### Scalar multiplication")

vector = np.array([st.number_input("X1",value=1.0,min_value=-5.0,max_value=5.0,step=0.5,key='scaled_X1'),st.number_input("Y1",value=2.0,min_value=-5.0,max_value=5.0,step=0.5,key='scaled_y1')])
st.write(f"**Scaled vector :**[{2*vector[0]} {2*vector[1]}]")

fig = utils.visualize_vector(v2=2*vector,v=vector,title="Scalar multiplication",grid_length=10)
st.pyplot(fig=fig)

st.markdown("### Vector Addition")



v1 = np.array([st.number_input("X1",value=1.0,min_value=-5.0,max_value=5.0,step=0.5,key='add_x1'),st.number_input("Y1",value=2.0,min_value=-5.0,max_value=5.0,step=0.5,key='add_y1')])
v2 = np.array([st.number_input("X2",value=-2.0,min_value=-5.0,max_value=5.0,step=0.5,key='add_x2'),st.number_input("Y2",value=3.0,min_value=-5.0,max_value=5.0,step=0.5,key='add_y2')])
v3 = v1 + v2
st.write(f"**V3 :**[{v3[0]} {v2[1]}]")

fig = utils.visualize_vector(v1=v1,v2=v2,v3=v3,title="Vector Addition",grid_length=10)
st.pyplot(fig=fig)

