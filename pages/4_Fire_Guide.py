import streamlit as st
import streamlit_shadcn_ui as ui
from lib.theme import inject_theme
from datetime import datetime

st.set_page_config(
    page_title="Fire Up - Fire Guide",
    page_icon="assets/Fire-Up-App-Logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)
inject_theme()

st.subheader("Fire Guide")
st.caption("Stay ready: essentials, preparation, checklists, and resources")

# --- Quick-Start Panel (kept & expanded) ---
st.markdown("""
<div class="panel">
  <div><strong>What You Must Know (Default)</strong></div>
</div>
""", unsafe_allow_html=True)
st.write("")
st.markdown(
    """
- **Have a Go-Bag**: water, N95 masks, flashlight, meds, chargers, copies of IDs.
- **Know Your Exits**: two+ evacuation routes; practice with family.
- **Defensible Space**: clear dry brush/leaves 30 ft from home.
- **Air Quality**: monitor AQI; seal windows; use HEPA if available.
- **Communication Plan**: emergency contacts and meetup point.
"""
)

# Quick resource buttons
qcols = st.columns(4)
with qcols[0]:
    ui.button("Ready.gov Wildfires", key="readygov_btn")
    st.markdown("[Open →](https://www.ready.gov/wildfires)")
with qcols[1]:
    ui.button("CAL FIRE (CA)", key="calfire_btn")
    st.markdown("[Open →](https://www.fire.ca.gov/)")
with qcols[2]:
    ui.button("AirNow AQI", key="airnow_btn")
    st.markdown("[Open →](https://www.airnow.gov/)")
with qcols[3]:
    ui.button("NWS Fire Weather", key="nws_btn")
    st.markdown("[Open →](https://www.weather.gov/fire/)")

st.divider()

# --- Helper: a small gallery function ---
def img_card(url, caption=""):
    st.image(url, use_container_width=True, caption=caption)

# --- Tabs for sections ---
tabs = st.tabs([
    "🏠 Preparation",
    "🔥 During a Fire",
    "🧭 Evacuation & Routes",
    "😷 Air Quality & Masks",
    "🛠️ Home Hardening",
    "🐾 Pets & Livestock",
    "🔌 Power & Generators",
    "📄 Insurance & Docs",
    "🧠 Mental Health",
    "🤝 Community & Volunteering",
    "📚 Glossary & Myths"
])

# 1) Preparation
with tabs[0]:
    st.header("Preparation")
    st.markdown("""
- **Build your Go-Bag** (3 days of supplies): water (4 L/person/day), shelf-stable food, medications, first-aid kit, **N95** masks, flashlight/headlamp + batteries, power bank & cables, multi-tool, cash, copies of IDs/insurance.
- **Family Communication Plan**: out-of-area contact, text-first policy, meeting spot, map of two+ exits.
- **Defensible Space**: 0–5 ft ember-resistant zone, 5–30 ft lean/clean/green, trim branches, clear gutters.
- **Stay Informed**: sign up for local alerts, follow **NWS**, **CAL FIRE**, county OES, and scanner apps (where legal).
- **Practice**: 10-minute evacuation drill, pet carriers ready, garage manual release tested.
""")

    # Downloadable checklist
    go_bag = """Go-Bag Checklist
- Water (4 L/person/day)
- Shelf-stable food (3 days)
- Medications + copies of prescriptions
- First-aid kit
- N95 masks (NIOSH)
- Flashlight/headlamp + batteries
- Power bank + cables
- Multi-tool / duct tape
- Cash (small bills)
- Copies of IDs, insurance, titles, medical records
- Change of clothes + sturdy shoes
- Hygiene items & sanitizers
- Pet supplies (carrier, food, meds)
"""
    st.download_button("Download Go-Bag Checklist (.txt)", go_bag, file_name="go_bag_checklist.txt")

    c1, c2, c3 = st.columns(3)
    with c1:
        img_card(
            "https://aarp.widen.net/content/waj0amhnsw/jpeg/20250320-.org-disaster-go-bag_-121_v2_D-curved.jpg?crop=true&anchor=84,65&q=80&color=ffffffff&u=lywnjt&w=1841&h=1058",
            "Example go-bag (credit: AARP)"
        )
    with c2:
        img_card(
            "https://protectivehealthgear.com/cdn/shop/files/N95-6150-Side_Shadow.png?v=1687904352&width=1000",
            "NIOSH-approved N95 type (credit: Protective Health Gear)"
        )
    with c3:
        img_card(
            "https://tcfswg.org/wp-content/uploads/2022/11/5C_graphic-defensible-space-1024x513.png",
            "Defensible space zones (credit: Tri-County FireSafe Working Group)"
        )

    st.markdown("""
**More:**  
- [Ready.gov: Wildfires](https://www.ready.gov/wildfires)  
- [CAL FIRE: Ready for Wildfire](https://www.readyforwildfire.org/)  
- [NFPA Home Ignition Zone](https://www.nfpa.org/hiz)
""")

# 2) During a Fire
with tabs[1]:
    st.header("During a Fire")
    st.markdown("""
- **Leave early if told to evacuate.** Do not wait—roads can close quickly.
- Wear long sleeves, long pants, sturdy shoes; bring **N95** for smoke; keep windows up in the car.
- **Inside**: keep doors/windows closed, run HEPA/clean room if safe; fill the car and position it outward.
- **Outside**: avoid canyons/ridges (chimney effect). Stay low if smoke thickens.
- **If trapped**: shelter in a cleared area or vehicle, keep engine running intermittently, call 911 with precise location.
""")
    st.markdown("""
**Resources:**  
- [Red Cross: Wildfire Safety](https://www.redcross.org/get-help/how-to-prepare-for-emergencies/types-of-emergencies/wildfire.html)  
- [USFS Fire Safety](https://www.fs.usda.gov/managing-land/fire)
""")
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/26/Wildfire_in_California.jpg", use_container_width=True, caption="Active wildfire (credit: Wikimedia Commons)")

# 3) Evacuation & Routes
with tabs[2]:
    st.header("Evacuation & Routes")
    st.markdown("""
- **Two+ ways out** from home, school, work. Keep a paper map.
- **Car ready**: half-tank minimum during peak season.
- **Go/No-Go list**: If **any** of these trip, **GO** — official evacuation order, fire within ~2 miles with wind toward you, **AQI > 300**, power loss + cellular outage.
- **Navigation**: Use your local county GIS/alerts; GPS can be wrong in closures. Listen to AM/FM emergency radio.
""")

    gonogo = """Go / No-Go Trigger List
- Official evacuation ORDER or WARNING
- Fire within ~2 miles and upwind toward your location
- AQI > 300 (Hazardous) + visible embers/ash
- Power outage + cell outage during red flag
- One primary route closed; smoke rapidly increasing
"""
    st.download_button("Download Go/No-Go Triggers (.txt)", gonogo, file_name="go_no_go_triggers.txt")

    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/3a/Evacuation_route_sign.jpg", use_container_width=True, caption="Know posted evacuation routes (credit: Wikimedia Commons)")
    with c2:
        st.markdown("""
**Helpful Links**  
- [Caltrans QuickMap (Roads)](https://quickmap.dot.ca.gov/)  
- [InciWeb (Incidents)](https://inciweb.nwcg.gov/)  
- [NIFC National Fire Maps](https://www.nifc.gov/fire-information)  
- Check your **County OES** / Sheriff alerts.
""")

# 4) Air Quality & Masks
with tabs[3]:
    st.header("Air Quality & Masks")
    st.markdown("""
- **Check AQI** via [AirNow](https://www.airnow.gov/) or local sensors (e.g., PurpleAir).  
- **N95/KN95** help reduce particulate inhalation; ensure proper fit.  
- Create a **clean room**: close windows, run HEPA purifier, seal gaps.  
- **In cars**: set to recirculate; consider a cabin filter upgrade (if supported).
""")
    c1, c2 = st.columns(2)
    with c1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/0c/AirNow_logo.png", use_container_width=True, caption="AirNow")
    with c2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/Purpleair_sensors.jpg", use_container_width=True, caption="Low-cost AQ sensors (credit: Wikimedia Commons)")
    st.markdown("""
**More:**  
- [EPA: Wildfire Smoke & Health](https://www.epa.gov/wildfire-smoke)  
- [CDC: Wildfire Smoke](https://www.cdc.gov/air/wildfires/index.html)
""")

# 5) Home Hardening
with tabs[4]:
    st.header("Home Hardening")
    st.markdown("""
- **Roof**: Class-A fire-rated; repair/replace wooden shakes.
- **Vents**: 1/8" metal mesh ember screens; baffled vents if possible.
- **Gutters & Decks**: clear debris; use non-combustible materials.
- **Windows**: dual-pane tempered; close curtains (radiant heat barrier).
- **0–5 ft**: non-combustible ground cover (gravel/stone), move mulch & firewood away.
""")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/09/House_Vent.jpg", use_container_width=True, caption="Vents: ember-resistant screens")
    with c2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/22/Double_glazed_window.jpg", use_container_width=True, caption="Dual-pane tempered windows")
    with c3:
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Metal_roof.jpg", use_container_width=True, caption="Class-A roofing example")
    st.markdown("""
**More:**  
- [IBHS Wildfire Prepared Home](https://ibhs.org/wildfire/)  
- [NFPA: Hardening Your Home](https://www.nfpa.org/hiz)
""")

# 6) Pets & Livestock
with tabs[5]:
    st.header("Pets & Livestock")
    st.markdown("""
- **Carriers & leashes** labeled with your contact info; practice loading quickly.
- Pet go-bag: food, water, meds, vet records, litter/liners, toys/blanket.
- Livestock: pre-arranged evacuation sites; trailer fuel & spares ready; halters tagged.
""")
    st.markdown("""
**More:**  
- [Ready.gov: Pets](https://www.ready.gov/pets)  
- Local animal services / fairgrounds for staging.
""")
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6e/Cat_in_carrier.jpg", use_container_width=True, caption="Carrier prep saves time (credit: Wikimedia Commons)")

# 7) Power & Generators
with tabs[6]:
    st.header("Power & Generators")
    st.markdown("""
- **PSPS / Outages**: keep flashlights & battery banks charged; gas up early.
- **Generators**: outdoor-only (CO risk), away from openings; transfer switch for home circuits.
- **Fuel safety**: stored in approved containers; keep a fire extinguisher (ABC) nearby.
""")
    st.markdown("""
**More:**  
- Check your utility’s **PSPS** portal (e.g., PG&E/SoCalEdison).
""")
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/62/Portable_generator.jpg", use_container_width=True, caption="Portable generator (credit: Wikimedia Commons)")

# 8) Insurance & Docs
with tabs[7]:
    st.header("Insurance & Documentation")
    st.markdown(f"""
- **Photo inventory** of rooms, serial numbers, receipts; store in cloud & on a USB in your go-bag.
- **Scan** IDs, titles, insurance, medical records; share with trusted contact.
- Review **Coverage A/B/C/D** (dwelling/other/personal/use loss); check ALE and code-upgrade riders.
- Timestamped updates help during claims (updated: {datetime.now():%Y-%m-%d}).
""")
    st.markdown("""
**More:**  
- Contact your insurer for a **Wildfire Endorsement** or FAIR plan (availability varies).
""")

# 9) Mental Health
with tabs[8]:
    st.header("Mental Health")
    st.markdown("""
- Disasters are stressful—normalize feelings. Use a **check-in plan** with family.
- Share age-appropriate info with kids; keep routines when possible.
- Use breathing apps, journaling, short walks indoors if smoke is severe.
""")
    st.markdown("""
**Resources:**  
- [SAMHSA Disaster Distress Helpline](https://www.samhsa.gov/find-help/disaster-distress-helpline)  
- Local county mental health services / school counselors.
""")

# 10) Community & Volunteering
with tabs[9]:
    st.header("Community & Volunteering")
    st.markdown("""
- Join **Fire Safe Councils** / **CERT**; help neighbors with drills & defensible space.
- Map hydrants, draft water sources, and vulnerable neighbors (privacy-aware).
- Donate to vetted local orgs; track needs lists from shelters.
""")
    st.markdown("""
**Get Involved:**  
- [Ready.gov CERT](https://www.ready.gov/cert)  
- Your **County Fire Safe Council** or **COAD/VOAD** network.
""")

# 11) Glossary & Myths
with tabs[10]:
    st.header("Glossary & Myths")
    st.markdown("""
**Glossary**  
- **Red Flag Warning**: critical fire-weather conditions expected now/soon.  
- **HIZ (Home Ignition Zone)**: the area around a structure that determines ignition risk.  
- **Spotting**: embers igniting fires ahead of the main front.  
- **PSPS**: Public Safety Power Shutoff to reduce ignition risk during extreme conditions.

**Myths**  
- “Green plants won’t burn.” ➜ **Myth**. Low moisture + wind = burn potential.  
- “If it’s smoky, the fire is far.” ➜ **Myth**. Smoke travel depends on wind/terrain; visibility can fool you.  
- “I’ll get a text if I must leave.” ➜ **Myth**. Don’t rely on one alert channel; use multiple.
""")

st.divider()

# --- Share & Footer (kept structure) ---
cols2 = st.columns([0.7, 0.3])
with cols2[1]:
    ui.button("Share", key="share_btn")

st.caption("Sources include Ready.gov, CAL FIRE, NWS, EPA AirNow, NFPA, IBHS, Red Cross, USFS. Always follow local authorities’ instructions.")
