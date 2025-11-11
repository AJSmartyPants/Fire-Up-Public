import streamlit as st
import pathlib
from lib.theme import inject_theme

APP_PATH = "https://fire-up.streamlit.app/" #change this url to a local one if you plan to test my code on your own machine
st.set_page_config(
    page_title="Fire Up",
    page_icon="assets/Fire-Up-App-Logo.jpeg",
    layout="wide",
    initial_sidebar_state="auto",
)

inject_theme()

st.markdown(
    """
    <div class="panel" style="margin-top:8px;">
      <div class="hero-title"><strong>Welcome to </strong> Fire Up!</div>
      <div class="hero-sub">Dynamic Prediction • Live Monitoring • Evacuation Routes • Wildfire Education • Emergency Contact</div>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    ("Home", "🔥", "pages/1_🔥 Home.py", "Home"),
    ("Detections", "📡", "pages/2_📡 Detections.py", "Detections"),
    ("Evacuation", "🧭", "pages/3_🧭 Evacuation.py", "Evacuation"),
    ("Fire Guide", "📚", "pages/4_📚 Fire_Guide.py", "Fire_Guide"),
    ("SOS", "🚨", "pages/5_🚨 SOS.py", "SOS"),
    ("About", "ℹ️", "pages/6_ℹ️ About.py", "About"),
]
#3x2 grid of cells
st.markdown(
    """
    <style>
      .grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 32px;
        margin-top: 32px;
      }
      .grid-tile {
        background: #fff;
        border-radius: 20px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 220px;
        transition: transform .15s ease, box-shadow .15s ease;
        text-align: center;
        cursor: pointer;
      }
      .grid-tile:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
      }
      .grid-emoji {
        font-size: 60px;
        margin-bottom: 14px;
      }
      .grid-label {
        font-size: 20px;
        font-weight: 700;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown("<div class='grid'>", unsafe_allow_html=True)

    rows = [st.columns(3)] 
    for i, (label, emoji, path, slug) in enumerate(PAGES[:3]):
        col = rows[0][i % 3]
        href = f"{APP_PATH}{slug}"
        with col:
            st.markdown(
                f"""
                <a href="{href}" target="_self" style="text-decoration:none; color:black; cursor:pointer;">
                  <div class='grid-tile'>
                    <div class='grid-emoji'>{emoji}</div>
                    <div class='grid-label'>{label}</div>
                  </div>
                </a>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    rows2 = [st.columns(3)]
    for i, (label, emoji, path, slug) in enumerate(PAGES[3:]):
        col = rows2[0][i % 3]
        href = f"{APP_PATH}{slug}"
        with col:
            st.markdown(
                f"""
                <a href="{href}" target="_self" style="text-decoration:none; color:black; cursor:pointer;">
                  <div class='grid-tile'>
                    <div class='grid-emoji'>{emoji}</div>
                    <div class='grid-label'>{label}</div>
                  </div>
                </a>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


st.caption("Tip: Click a tile to explore. You can also use the default hamburger on the left.")
