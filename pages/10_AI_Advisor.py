import streamlit as st
from src.tyrewear_app import ai_advisor
st.set_page_config(page_title="AI Advisor", page_icon="TW", layout="wide")
ai_advisor()
