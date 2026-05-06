import streamlit as st
from src.tyrewear_app import correlation
st.set_page_config(page_title="Correlation Analysis", page_icon="TW", layout="wide")
correlation()
