import streamlit as st
from src.tyrewear_app import geo_map
st.set_page_config(page_title="Geo Map", page_icon="TW", layout="wide")
geo_map()
