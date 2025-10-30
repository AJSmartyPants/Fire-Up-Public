import streamlit as st
import streamlit_shadcn_ui as ui
from lib.theme import inject_theme
from datetime import datetime
import webbrowser

st.set_page_config(
    page_title="Fire Up - SOS",
    page_icon="assets/Fire-Up-App-Logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)
inject_theme()

st.subheader("SOS")
st.caption("Share your location and status with contacts; quick access to 911")

# --- Inputs ---
name = st.text_input("Your name", key="sos_name")

# Message mode: Type vs Template
mode = st.radio(
    "How do you want to set your status message?",
    options=["Type my own", "Choose a pre-written scenario"],
    horizontal=True,
    key="sos_mode"
)

# Pre-written wildfire scenarios
SCENARIOS = {
    "Evacuating now (ordered)": (
        "EVACUATING NOW — evacuation ORDER received.\n"
        "Name: {name}\n"
        "Status: Leaving immediately by car on primary route.\n"
        "People/Pets: [# adults], [# kids], [pets]\n"
        "Needs: None right now. Will confirm arrival at safe location.\n"
        "Notes: Roads may be congested; will avoid closed areas."
    ),
    "Sheltering in place (heavy smoke)": (
        "SHELTERING IN PLACE — heavy smoke, no active flames nearby.\n"
        "Name: {name}\n"
        "Status: Sealed indoors, running HEPA/clean room.\n"
        "People/Pets: [list]\n"
        "Needs: N95 masks / air purifier filters if available.\n"
        "Notes: Monitoring alerts; ready to evacuate if ordered."
    ),
    "Trapped / need rescue": (
        "NEED RESCUE — unable to evacuate safely.\n"
        "Name: {name}\n"
        "Status: Taking shelter in cleared area/vehicle.\n"
        "People/Pets: [list]\n"
        "Injuries: [describe or 'none']\n"
        "Last Known Location: [address/cross-streets/GPS]\n"
        "Notes: Visible flames/embers nearby. Requesting immediate assistance."
    ),
    "Missing check-in / limited comms": (
        "CHECK-IN — limited cell/data; sending status.\n"
        "Name: {name}\n"
        "Status: Safe for now; power/cell intermittent.\n"
        "People/Pets: [list]\n"
        "Needs: None currently. Will check in every 2 hours if possible."
    ),
    "Medical help needed": (
        "MEDICAL HELP NEEDED — wildfire related.\n"
        "Name: {name}\n"
        "Symptoms/Injury: [describe]\n"
        "Current Location: [address/cross-streets/GPS]\n"
        "Notes: Smoke exposure level [mild/moderate/severe]. Need guidance/transport."
    ),
    "Power & cell outage — safe": (
        "SAFE BUT WITHOUT POWER/CELL.\n"
        "Name: {name}\n"
        "Status: Shelter safe; no immediate fire threat.\n"
        "People/Pets: [list]\n"
        "Needs: Ice/charging/medical devices support if available.\n"
        "Plan: Will relocate if conditions worsen."
    ),
}

# --- Determine initial text value safely ---
if "sos_status_text" not in st.session_state:
    st.session_state["sos_status_text"] = "I need help. Here's my status..."

if mode == "Choose a pre-written scenario":
    scenario_key = st.selectbox(
        "Select a wildfire scenario",
        options=list(SCENARIOS.keys()),
        index=0,
        key="sos_scenario"
    )
    templated = SCENARIOS[scenario_key].format(name=name or "[Your Name]")
    if (
        st.session_state.get("last_applied_scenario") != scenario_key
        or st.session_state.get("last_mode") != mode
    ):
        st.session_state["sos_status_text"] = templated
        st.session_state["last_applied_scenario"] = scenario_key

status_text = st.text_area(
    "Status message",
    height=160,
    key="sos_status_text"
)

contacts = st.text_input(
    "Contacts (comma-separated emails/phone numbers)",
    key="sos_contacts",
    placeholder="e.g., mom@example.com, +1-555-123-4567"
)

st.write("")

# --- Action buttons ---
colA, colB, colC = st.columns([0.4, 0.35, 0.25])
with colA:
    ui.button("Send My Location & Status", key="sos_send")
with colB:
    ui.button("Copy Message to Clipboard", key="copy_msg")
with colC:
    ui.button("Call 911", key="call_911", variant="destructive")

# --- Preview ---
st.markdown("---")
st.markdown("**Preview**")
with st.container():
    st.code(
        f"Name: {name or '[Your Name]'}\n"
        f"Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Contacts: {contacts or '[none]'}\n\n"
        f"{status_text}",
        language="markdown"
    )

# --- External Resource Button ---
st.markdown("---")
st.markdown("**Get Involved:**")
if st.button("Contribute to Fire-Adapted Communities"):
    webbrowser.open_new_tab("https://fireadapted.org/resource-type/community-members/")

# Track last mode
st.session_state["last_mode"] = mode

st.markdown("</div>", unsafe_allow_html=True)
