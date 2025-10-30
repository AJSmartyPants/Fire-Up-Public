import streamlit as st

def inject_theme():
    CSS = """
    <style>
      :root {
        /* Warm LIGHT palette */
        --bg: #fffdf7;            /* paper white */
        --text: #151515;          /* readable dark */
        --muted: #5a5a5a;         /* captions */
        --panel: #ffffff;         /* cards */
        --border: #f1eadf;        /* soft border */
        --shadow: 0 10px 30px rgba(0,0,0,.07);

        /* Wildfire accents */
        --accent: #e53935;        /* red */
        --accent-2: #ff8f00;      /* amber */
        --accent-3: #ff7043;      /* deep orange */
        --accent-grad: linear-gradient(135deg, #ff8f00, #ff7043 50%, #e53935);

        --radius: 16px;
      }

      html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text) !important; }
      .panel { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow); padding: 16px; }
      .brand { font-weight: 800; letter-spacing: .2px; }
      .badge { padding: 6px 10px; border-radius: 999px; font-size: 12px; border: 1px solid #f3e7d6; color: #7a6a54; background: #fff7ec; }

      /* Header */
      .fireup-header { position: sticky; top: 0; z-index: 9999; display: flex; align-items: center; gap: 14px; padding: 14px 16px; background: rgba(255,253,247,.85); backdrop-filter: blur(6px); border-bottom: 1px solid var(--border); }
      .fireup-title { font-size: 20px; }
      .fireup-sub { font-size: 13px; color: var(--muted); }

      /* Hero */
      .hero { margin: 10px 8px; background: #fff; }
      .hero-title { font-size: 26px; font-weight: 800; margin-bottom: 6px; background: var(--accent-grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
      .hero-sub { color: var(--muted); margin-bottom: 10px; }
      .hero-cta { font-size: 13px; color: #7a6a54; }

      /* Map placeholders */
      .mapph { height: 420px; border-radius: 14px; border: 2px dashed #ffe0b2; background: repeating-linear-gradient(45deg, #fffaf3, #fffaf3 10px, #fff5e6 10px, #fff5e6 20px); display:grid; place-items:center; color:#b07c37; }
      /* Make the single visible hamburger pretty */
      #hamb_wrap button { width: 44px; height: 44px; border-radius: 12px; background: var(--panel); border: 1px solid var(--border); box-shadow: var(--shadow); font-size: 18px; line-height: 1; }
    </style>
    """
    st.markdown(CSS, unsafe_allow_html=True)