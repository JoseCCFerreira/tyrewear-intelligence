import streamlit as st
from src.tyrewear_app import tread_depth
st.set_page_config(page_title="Tread Depth Analysis", page_icon="TW", layout="wide")
tread_depth()
