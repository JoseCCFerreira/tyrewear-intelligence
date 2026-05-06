from __future__ import annotations

import streamlit as st

from src.tyrewear_app import overview


st.set_page_config(page_title="TyreWear Intelligence", page_icon="TW", layout="wide")
overview()
