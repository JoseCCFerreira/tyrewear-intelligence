import streamlit as st
from src.tyrewear_app import statistical_tests
st.set_page_config(page_title="Statistical Tests", page_icon="TW", layout="wide")
statistical_tests()
