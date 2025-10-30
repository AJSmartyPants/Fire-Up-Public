import streamlit as st
import pathlib
from lib.theme import inject_theme

APP_PATH = "http://localhost:8501/"
st.set_page_config(
    page_title="Fire Up — Welcome",
    page_icon="🔥",
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
    ("Home", "🔥", "pages/1_Home.py"),
    ("Detections", "📡", "pages/2_Detections.py"),
    ("Evacuation", "🧭", "pages/3_Evacuation.py"),
    ("Fire Guide", "📚", "pages/4_Fire_Guide.py"),
    ("SOS", "🚨", "pages/5_SOS.py"),
    ("About", "ℹ️", "pages/6_About.py"),
]

# Grid styling with large tiles that feel like cells, not buttons
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

    rows = [st.columns(3)]  # 3×2 grid
    for i, (label, emoji, path) in enumerate(PAGES[:3]):
        col = rows[0][i % 3]
        slug = pathlib.Path(path).stem
        if "_" in slug:
            slug = slug.split("_", 1)[1]
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
    for i, (label, emoji, path) in enumerate(PAGES[3:]):
        col = rows2[0][i % 3]
        slug = pathlib.Path(path).stem
        if "_" in slug:
            slug = slug.split("_", 1)[1]
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
