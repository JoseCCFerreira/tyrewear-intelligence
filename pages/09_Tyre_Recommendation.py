import streamlit as st
from src.tyrewear_app import recommendation
st.set_page_config(page_title="Tyre Recommendation", page_icon="TW", layout="wide")
recommendation()
