import streamlit as st
import pandas as pd
import numpy as np

st.title("predicting loan payback")

st.text("Welcome!")


# button for start predicting
if st.button("start"):
    try:
        st.switch_page("pages/collect_data.py")
    except Exception as e:
        st.error(f"Error: {e}")
