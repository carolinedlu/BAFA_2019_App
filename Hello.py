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
    ### The pages within this App tell the story about how I used some libraries in Python :snake: to aquire data, analyze it and implement a model on it.  
    **👈 Select a Page for you to explore!
    ### Sections to explore in this App  
    - About BAFA :football:
    - Data Collection and Questions :computer:
    - Findings :mag_right: :bar_chart:
    - Models :chart_with_upwards_trend:

    ### The overall approach was
    - Webscrapping and API's
    - Data Integration 
    - Data Analysis
    - ML
"""
)

st.markdown("## About me")

st.markdown(
    """
    #### PhD Candidate on management science at Strathclyde University Business School. Currently working on bi-level optimization. 
    ##### Interests:
    - Linear Optimization
    - Mixed Integer Linear Optimization
    - Discrete Network Optimization
    - Data Analysis  

"""
)

st.markdown(
    """ 
    ### Contact Details
    - e-mail : alexander_vindel@gmail.com
    - LinkedIn : [alexander-vindel](https://www.linkedin.com/in/alexander-vindel/)
    - GitHub : [@j-alex-vindel](https://github.com/j-alex-vindel)

    """)
