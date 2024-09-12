import streamlit as st

from pages.introduction import Introduction
from pages.analysis import Analysis
from pages.conclusions import Conclusion
from pages.dataset import Dataset
from pages.modeling import Modeling

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Introduction", "Dataset", "Data Analysis", "Modeling", "Conclusions"])

# Page Routing
if options == "Introduction":
    Introduction(st).render()
elif options == "Dataset":
    Dataset(st).render()
elif options == "Data Analysis":
    Analysis(st).render()
elif options == "Modeling":
    Modeling(st).render()
elif options == "Conclusions":
    Conclusion(st).render()