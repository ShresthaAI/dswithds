import streamlit as st
from utils.utils import load_page
st.set_page_config(page_title="DS with DS", layout="wide",initial_sidebar_state="collapsed")



st.sidebar.title("Navigation")
pages = {
    "Precision & Recall": "metrics/1_Precision_Recall",
    "Vectors": "linalg/Vectors",
    "About": "About"
}

choice = st.sidebar.selectbox("Go to", list(pages.keys()))

# Assuming load_page takes page name as input
load_page(pages[choice])

