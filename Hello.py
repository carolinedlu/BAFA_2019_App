import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Alex's SideProject",
    page_icon=":necktie:"
)

st.write("# Welcome to myPage! 👋")

st.sidebar.success("Select a page to visit.")

st.markdown(
    """
    Here I should write about the pages
    **👈 Select a Page for you to explore!
    ### Sections to explore in this App  
    - About BAFA :football:
    - Data Collection and Questions :computer:
    - Findings :mag_right: :bar_chart:
    - Models :chart_with_upwards_trend:
"""
)

st.markdown("## About me")

st.markdown(
    """
    ### Some details about me
    - Background
    - Github
    - LinkedIn

"""
)
