import streamlit as st
import streamlit_shadcn_ui as ui
import time, io, requests, numpy as np, cv2, json
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from pathlib import Path
from lib.theme import inject_theme
import tensorflow as tf

st.set_page_config(
    page_title="📡 Detection Page - Fire Up",
    page_icon="assets/Fire-Up-App-Logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)
inject_theme()
st_autorefresh(interval=5 * 1000, key="sensor_refresh")

st.subheader("Detections")
st.caption("Live sensor stream from Raspberry Pi and CNN identification of image input")

pi_url = st.text_input(
    "Enter your Raspberry Pi Flask URL",
    value=st.session_state.get("pi_url", ""),
    placeholder="http://192.168.1.50:8000"
)
st.session_state["pi_url"] = pi_url.strip()

if not pi_url:
    st.info("Start your Pi server and paste its URL above.")
    st.stop()

#setup
MODEL_PATH = Path("models/fire-classify.keras")
LABELS_PATH = Path("models/labels.json")

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    try:
        labels = json.loads(LABELS_PATH.read_text())
    except Exception:
        labels = ["nothing", "smoke", "fire"]
    return model, labels

def preprocess_for_model(bgr, model):
    ishape = model.input_shape
    if isinstance(ishape, list): 
        ishape = ishape[0]
    h, w = ishape[1], ishape[2]

    H, W = bgr.shape[:2]
    side = min(H, W)
    y0 = (H - side) // 2
    x0 = (W - side) // 2
    crop = bgr[y0:y0+side, x0:x0+side]
    resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)[None, ...]

def fmt(val, suffix=""):
    if val is None:
        return "—"
    try:
        return f"{val:.1f}{suffix}"
    except Exception:
        return str(val)

def brightness_label(pct):
    if pct is None:
        return "—"
    if pct < 20:  return "Very Low"
    if pct < 40:  return "Low"
    if pct < 70:  return "Medium"
    if pct < 90:  return "High"
    return "Very High"

def last_hms(iso_str):
    if not iso_str:
        return "—"
    try:
        t = datetime.fromisoformat(iso_str)
        return t.strftime("%H:%M:%S")
    except Exception:
        return iso_str

c1, c2 = st.columns([0.6, 0.4])

#camera & inference
with c1:
    st.markdown("""
    <div class="panel">
    <div><strong>Camera Feed</strong></div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown(
        f'<img src="{pi_url}/mjpeg" style="width:100%;border-radius:8px;border:1px solid #444;" />',
        unsafe_allow_html=True
    )
    st.divider()
    st.subheader("On-Demand Inference")

    model_ok = True
    try:
        model, labels = load_model_and_labels()
    except Exception as e:
        model_ok = False
        st.error(f"Model not loaded: {e}\nExpected at: {MODEL_PATH}")
        labels = ["nothing", "smoke", "fire"]

    if st.button("Analyze Latest Frame"):
        try:
            jpg = requests.get(f"{pi_url}/frame.jpg", timeout=5).content
            arr = np.frombuffer(jpg, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption="Snapshot used for inference", use_container_width=True)

            if model_ok:
                x = preprocess_for_model(bgr, model)
                probs = model.predict(x, verbose=0)[0]
                idx = int(np.argmax(probs))
                label = labels[idx] if idx < len(labels) else f"class_{idx}"
                conf = float(probs[idx])

                ui.metric_card(
                    title="Detection Result",
                    content=f"{label.upper()} ({conf*100:.1f}%)",
                    description="Based on latest frame",
                    key="pd_detection",
                )
                st.bar_chart({labels[i]: float(probs[i]) for i in range(len(probs))})

        except Exception as e:
            st.error(f"Failed to fetch or analyze frame: {e}")

#sensor readings
with c2:
    st.markdown("""
    <div class="panel">
    <div><strong>Sensor Readings</strong></div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    sens = None
    try:
        r = requests.get(f"{pi_url}/sensors", timeout=3)
        sens = r.json()
    except Exception as e:
        st.warning(f"Could not fetch sensors: {e}")

    temp_c         = sens.get("temp_c") if sens else None
    humidity_pct   = sens.get("humidity_pct") if sens else None
    brightness_pct = sens.get("brightness_lux") if sens else None
    fps            = sens.get("fps") if sens else None
    pkt_time       = sens.get("time") if sens else None

    ui.metric_card(
        title="Temperature",
        content=fmt(temp_c, " °C"),
        description="DHT20 (with −10°C adjustment on Pi)",
        key="pd_temp",
    )
    ui.metric_card(
        title="Humidity",
        content=fmt(humidity_pct, " %"),
        description="DHT20",
        key="pd_hum",
    )
    ui.metric_card(
        title="Brightness",
        content=brightness_label(brightness_pct),
        description=f"{fmt(brightness_pct, ' %')} (LDR)",
        key="pd_brightness",
    )
    ui.metric_card(
        title="Camera FPS",
        content=fmt(fps, " FPS"),
        description="From Pi MJPEG loop",
        key="pd_fps",
    )

#drone placeholder
st.subheader("Sample Drone Integration")
st.caption("Preview how users would track drone swarms and incidents")

cc1, cc2 = st.columns([0.64, 0.36])
with cc1:
    st.markdown("""
    <div class="panel">
    <div><strong>Mission Map (placeholder)</strong></div>
    <div class="subtle">Track drone positions, tasking, video feeds overlays.</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("<div class='mapph' style='height:430px'>[Drone mission map placeholder]</div>", unsafe_allow_html=True)
with cc2:
    st.markdown("""
    <div class="panel">
    <div><strong>Fleet & Telemetry</strong></div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.dataframe({
        "Drone": ["ACERO-01", "ACERO-02", "Scout-Alpha"],
        "Battery": ["82%", "64%", "91%"],
        "Link": ["OK", "OK", "WARN"],
        "Task": ["Perimeter scan", "IR hotspot", "Resupply"],
    })
    ui.button("View Drones Near Areas with Fire Risk", key="drone_risk")
    ui.button("Assign Drone Task (requires admin access)", key="drone_task")
