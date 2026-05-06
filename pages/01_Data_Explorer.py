import streamlit as st
from src.tyrewear_app import data_explorer
st.set_page_config(page_title="Data Explorer", page_icon="TW", layout="wide")
data_explorer()
