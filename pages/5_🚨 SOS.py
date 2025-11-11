import os
import csv
from datetime import datetime

import streamlit as st
import streamlit_shadcn_ui as ui
import streamlit_js_eval as st_js

from lib.theme import inject_theme

st.set_page_config(
    page_title="🚨 SOS - Fire Up",
    page_icon="assets/Fire-Up-App-Logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_theme()

st.subheader("SOS")
st.caption("Share your live location and status with contacts; quick access to 911")

LOG_PATH = "data/sos_log.csv"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def get_location():
    """Get browser geolocation (if allowed). Returns (lat, lon, accuracy_m) or (None, None, None)."""
    try:
        loc = st_js.get_geolocation()
        if not loc or "coords" not in loc:
            return None, None, None
        c = loc["coords"]
        return float(c.get("latitude")), float(c.get("longitude")), float(c.get("accuracy"))
    except Exception:
        return None, None, None

def maps_link(lat, lon):
    return f"https://maps.google.com/?q={lat:.6f},{lon:.6f}"

def build_message(name, status, lat, lon, acc_m):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    coords_line = (
        f"Location: {lat:.6f}, {lon:.6f} (±{int(acc_m)} m)\nMap: {maps_link(lat,lon)}"
        if lat is not None and lon is not None else
        "Location: not granted (please share your location if safe)"
    )
    return (
        f"Name: {name or '[Your Name]'}\n"
        f"Timestamp: {ts}\n"
        f"{coords_line}\n\n"
        f"{status.strip()}"
    )

def parse_contacts(raw):
    """Return list of cleaned contacts. Accepts emails or phone numbers separated by commas."""
    if not raw:
        return []
    out = []
    for token in raw.split(","):
        t = token.strip()
        if t:
            out.append(t)
    return out

def contact_actions(contact, msg):
    """Return label + (url_label, url) pairs for this contact."""
    if "@" in contact: 
        subj = "Fire Up SOS Status"
        url = (
            "mailto:"
            f"{contact}"
            f"?subject={st_js.quote(subj)}"
            f"&body={st_js.quote(msg)}"
        )
        return contact, [("Email", url)]
    digits = contact
    sms_url = f"sms:{digits}?&body={st_js.quote(msg)}"
    wa_url  = f"https://wa.me/{''.join([c for c in digits if c.isdigit()])}?text={st_js.quote(msg)}"
    tel_url = f"tel:{digits}"
    return contact, [("SMS", sms_url), ("WhatsApp", wa_url), ("Call", tel_url)]

def log_sos(row_dict):
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=[
            "timestamp","name","lat","lon","accuracy_m","contacts","message"
        ])
        if write_header:
            wr.writeheader()
        wr.writerow(row_dict)

SCENARIOS = {
    "Evacuating now (ordered)": (
        "EVACUATING NOW — evacuation ORDER received.\n"
        "Status: Leaving immediately by car on primary route.\n"
        "People/Pets: [# adults], [# kids], [pets]\n"
        "Needs: None right now. Will confirm arrival at safe location.\n"
        "Notes: Roads may be congested; will avoid closed areas."
    ),
    "Sheltering in place (heavy smoke)": (
        "SHELTERING IN PLACE — heavy smoke, no active flames nearby.\n"
        "Status: Sealed indoors, running HEPA/clean room.\n"
        "People/Pets: [list]\n"
        "Needs: N95 masks / air purifier filters if available.\n"
        "Notes: Monitoring alerts; ready to evacuate if ordered."
    ),
    "Trapped / need rescue": (
        "NEED RESCUE — unable to evacuate safely.\n"
        "Status: Taking shelter in cleared area/vehicle.\n"
        "People/Pets: [list]\n"
        "Injuries: [describe or 'none']\n"
        "Last Known Location: [address/cross-streets/GPS]\n"
        "Notes: Visible flames/embers nearby. Requesting immediate assistance."
    ),
    "Missing check-in / limited comms": (
        "CHECK-IN — limited cell/data; sending status.\n"
        "Status: Safe for now; power/cell intermittent.\n"
        "People/Pets: [list]\n"
        "Needs: None currently. Will check in every 2 hours if possible."
    ),
    "Medical help needed": (
        "MEDICAL HELP NEEDED — wildfire related.\n"
        "Symptoms/Injury: [describe]\n"
        "Current Location: [address/cross-streets/GPS]\n"
        "Notes: Smoke exposure level [mild/moderate/severe]. Need guidance/transport."
    ),
    "Power & cell outage — safe": (
        "SAFE BUT WITHOUT POWER/CELL.\n"
        "Status: Shelter safe; no immediate fire threat.\n"
        "People/Pets: [list]\n"
        "Needs: Ice/charging/medical devices support if available.\n"
        "Plan: Will relocate if conditions worsen."
    ),
}

name = st.text_input("Your name", key="sos_name")

mode = st.radio(
    "How do you want to set your status message?",
    options=["Type my own", "Choose a pre-written scenario"],
    horizontal=True,
    key="sos_mode",
)

if "sos_status_text" not in st.session_state:
    st.session_state["sos_status_text"] = "I need help. Here's my status..."

if mode == "Choose a pre-written scenario":
    scenario_key = st.selectbox(
        "Select a wildfire scenario",
        options=list(SCENARIOS.keys()),
        index=0,
        key="sos_scenario",
    )
    if st.session_state.get("last_applied_scenario") != scenario_key:
        st.session_state["sos_status_text"] = SCENARIOS[scenario_key]
        st.session_state["last_applied_scenario"] = scenario_key

status_text = st.text_area("Status message", height=160, key="sos_status_text")

contacts_raw = st.text_input(
    "Contacts (comma-separated emails/phone numbers)",
    key="sos_contacts",
    placeholder="e.g., mom@example.com, +1-555-123-4567",
)

st.write("")

lat, lon, acc_m = get_location()
message = build_message(name, status_text, lat, lon, acc_m)

colA, colB, colC = st.columns([0.44, 0.36, 0.20])

with colA:
    if ui.button("Send My Location & Status", key="sos_send"):
        recips = parse_contacts(contacts_raw)
        if not recips:
            st.warning("Add at least one contact (email or phone) above.")
        else:
            st.success("Choose how to send to each contact:")
            for c in recips:
                label, actions = contact_actions(c, message)
                cols = st.columns(len(actions))
                for i, (alabel, url) in enumerate(actions):
                    with cols[i]:
                        st.link_button(f"{alabel}: {label}", url, use_container_width=True)

            log_sos({
                "timestamp": datetime.utcnow().isoformat(),
                "name": name,
                "lat": lat, "lon": lon, "accuracy_m": acc_m,
                "contacts": contacts_raw,
                "message": message,
            })

with colB:
    if ui.button("Copy Message to Clipboard", key="copy_msg"):
        try:
            st_js.eval_js(f"navigator.clipboard.writeText({st_js.escape_js(message)!r})")
            st.success("Message copied to clipboard.")
        except Exception:
            st.info("Couldn’t access clipboard. Use the copy button on the block below.")
            st.code(message)

with colC:
    st.link_button("📞 Call 911", "tel:911", use_container_width=True)

st.markdown("---")
st.markdown("**Preview**")
st.code(message)

st.markdown("---")
st.markdown("**Get Involved:**")
st.link_button(
    "Contribute to Fire-Adapted Communities",
    "https://fireadapted.org/resource-type/community-members/",
    use_container_width=False,
)

st.session_state["last_mode"] = mode
