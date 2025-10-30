import streamlit as st
import streamlit_shadcn_ui as ui
from lib.theme import inject_theme

st.set_page_config(
    page_title="Fire Up - About",
    page_icon="assets/Fire-Up-App-Logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)
inject_theme()

st.subheader("About")