import streamlit as st
import numpy as np
from utils import utils

def app():
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

    col1, col2 = st.columns([1, 2])
    with col1:
        vector_x1 = st.slider("X1", min_value=-5.0, max_value=5.0, value=1.0, step=0.5, key='x1')
        vector_y1 = st.slider("Y1", min_value=-5.0, max_value=5.0, value=1.0, step=0.5, key='y1')

        st.write(f"[{vector_x1}, {vector_y1}]")
    with col2:
        fig = utils.visualize_vector(title="Vector Visualization", show=False,
                                A=np.array([vector_x1, vector_y1]), colors=['red'])
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
    col1, col2 = st.columns([1, 2])

    with col1:
        norm_vector_x1 = st.slider("X1", min_value=-5.0, max_value=5.0, value=1.0, step=0.5, key='norm_x1')
        norm_vector_y1 = st.slider("Y1", min_value=-5.0, max_value=5.0, value=1.0, step=0.5, key='norm_y1')

        st.write(f"**Normalized vector:** [{norm_vector_x1}, {norm_vector_y1}]")

        norm = utils.normalize_vector(np.array([norm_vector_x1, norm_vector_y1]))
    with col2:
        fig = utils.visualize_vector(v=np.array([norm_vector_x1, norm_vector_y1]), norm=norm, 
                                title="Arrow representation of vector and normalized form")
        st.pyplot(fig=fig)

    st.markdown("## Vector Operations")
    st.markdown("### Scalar multiplication")
    col1, col2 = st.columns([1, 2])

    with col1:
        scaled_vector_x1 = st.slider("X1", min_value=-5.0, max_value=5.0, value=1.0, step=0.5, key='scaled_X1')
        scaled_vector_y1 = st.slider("Y1", min_value=-5.0, max_value=5.0, value=2.0, step=0.5, key='scaled_y1')

        st.write(f"**Scaled vector:** [{2 * scaled_vector_x1}, {2 * scaled_vector_y1}]")
    with col2:
        fig = utils.visualize_vector(v2=2 * np.array([scaled_vector_x1, scaled_vector_y1]), v=np.array([scaled_vector_x1, scaled_vector_y1]), 
                                title="Scalar multiplication", grid_size=10)
        st.pyplot(fig=fig)

    st.markdown("### Vector Addition")

    col1, col2 = st.columns([1, 2])
    with col1:
        add_vector_x1 = st.slider("X1", min_value=-5.0, max_value=5.0, value=1.0, step=0.5, key='add_x1')
        add_vector_y1 = st.slider("Y1", min_value=-5.0, max_value=5.0, value=2.0, step=0.5, key='add_y1')
        add_vector_x2 = st.slider("X2", min_value=-5.0, max_value=5.0, value=-2.0, step=0.5, key='add_x2')
        add_vector_y2 = st.slider("Y2", min_value=-5.0, max_value=5.0, value=3.0, step=0.5, key='add_y2')

        v1 = np.array([add_vector_x1, add_vector_y1])
        v2 = np.array([add_vector_x2, add_vector_y2])
        v3 = v1 + v2

        st.write(f"**V3:** [{v3[0]}, {v3[1]}]")
    
    with col2:

        fig = utils.visualize_vector(v1=v1, v2=v2, v3=v3, title="Vector Addition", grid_size=10)
        st.pyplot(fig=fig)

    st.markdown(
        """
        ## Linear Combination of vectors

    $$
    \mathbf{u} = c_1\mathbf{v}^1 + c_2\mathbf{v}^2 + ...+ c_n\mathbf{v}^n
    $$

    $ \mathbf{u} $ is called a linear combination of vectors $\mathbf{v}^1$,$\mathbf{v}^2$,...,$\mathbf{v}^n$ with weights $c_1$,$c_2$,...,$c_n$
    """
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        lin_comb_vector_x1 = st.slider("X1", min_value=-5.0, max_value=5.0, value=3.0, step=0.5, key='lin_x1')
        lin_comb_vector_y1 = st.slider("Y1", min_value=-5.0, max_value=5.0, value=1.0, step=0.5, key='lin_y1')
        lin_comb_vector_x2 = st.slider("X2", min_value=-5.0, max_value=5.0, value=-1.0, step=0.5, key='lin_x2')
        lin_comb_vector_y2 = st.slider("Y2", min_value=-5.0, max_value=5.0, value=2.0, step=0.5, key='lin_y2')
        lin_comb_weight_c1 = st.slider("c1", min_value=-2.0, max_value=2.0, value=1.2, step=0.1, key='lin_c1')
        lin_comb_weight_c2 = st.slider("c2", min_value=-2.0, max_value=2.0, value=1.3, step=0.1, key='lin_c2')

    v1 = np.array([lin_comb_vector_x1, lin_comb_vector_y1])
    v2 = np.array([lin_comb_vector_x2, lin_comb_vector_y2])
    weights = [lin_comb_weight_c1, lin_comb_weight_c2]

    z = weights[0] * v1 + weights[1] * v2
    with col2:
        fig = utils.visualize_vector(x=v1, y=v2, z=z, title="Linear combination of vectors", grid_size=8)
        st.pyplot(fig=fig)

    st.markdown(
        """
        ## Vector Spaces
        ### Span & Linear Independence

        **Linear dependence** :

        Any set of vectors is linearly dependent if one can be written as linear combination of others. 

        Otherwise they are linearly independent

    """
    )

    x = np.array([2, 3])
    y = np.array([3, 4.5])
    z = np.array([-1, -1.5])
    
    fig = utils.visualize_vector(x=x, y=y, z=z, title="Linearly Dependent Vectors")
    st.pyplot(fig=fig)

    x = np.array([2, 3])
    y = np.array([-3, -2])
    z = np.array([-1, 3])
    fig = utils.visualize_vector(x=x, y=y, z=z, title="Linearly Independent Vectors")
    st.pyplot(fig=fig)

    st.markdown(
        """### Basis Vectors"""
    )
    col1, col2 = st.columns([1, 2])
    with col1:

        basis_vector_x1 = st.slider("X1", min_value=-5.0, max_value=5.0, value=-1.0, step=0.5, key='basis_x1')
        basis_vector_y1 = st.slider("Y1", min_value=-5.0, max_value=5.0, value=1.0, step=0.5, key='basis_y1')
        basis_vector_x2 = st.slider("X2", min_value=-5.0, max_value=5.0, value=1.0, step=0.5, key="baxis_x2")
        basis_vector_y2 = st.slider("y2", min_value=-5.0, max_value=5.0, value=1.0, step=0.5, key="baxis_y2")
    with col2:
        fig = utils.visualize_vector(title="Change of bases",new_basis=np.array([[basis_vector_x1,basis_vector_y1],[basis_vector_x2,basis_vector_y2]]),A=np.array([2,1]),B=np.array([-2,1])) 
        if not fig:
            st.write("Error: Basis vectors are linearly dependent.")
        else:
            st.pyplot(fig=fig)