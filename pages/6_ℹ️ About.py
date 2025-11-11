import streamlit as st
import streamlit_shadcn_ui as ui
from lib.theme import inject_theme

st.set_page_config(
    page_title="ℹ️ About - Fire Up",
    page_icon="assets/Fire-Up-App-Logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)
inject_theme()

#sources
ME_PHOTO   = "assets/AnikaPicture.jpg"
APP_LOGO   = "assets/Fire-Up-App-Logo.jpeg"
VIDEO_URL  = "https://youtu.be/M6boFqd9h_E"
CONTACT_EMAIL = "anikajhaa8@gmail.com"

#about me
st.markdown("### 👩🏽‍💻 About the Developer")

col1, col2 = st.columns([0.7, 0.3], gap="medium")

with col1:
    st.markdown("""
**Hi, I'm Anika Jha!**  
I'm an aspiring changemaker and philomath. I love all things STEM! \n
~ Received **Diana Award**, **Technovation Girls** International Grand Prize & Technology award for 2 apps  \n~ **Ted Ed Student** Talks Speaker, **author** of 2 children's books and volume of poetry  \n~ Winner of global **Rube Goldberg Playground Design** Contest  \n~ 1st Place Winner **NASA Dream With Us Design** Challenge  \n~ Developed CubeSAT prototype & won international **StudentSAT** contest  \n~ Won International Mars Design Competition by the **Mars Society** \n
As someone living in Los Angeles, California, I was deeply affected by the Palisades Fire in early 2025. Seeing people suffer first-hand, being unable to step outside due to smoke, and feeling afraid and clueless about fires led me to research more about them and understand the problem. I found that: 
- Prediction tools are either static (only based on past data) or unavailable to the public 
- Our current detection methods can be slow and imprecise 
- There is no dedicated, accessible solution targeting all of these core issues while spreading awareness \n
This inspired me to create Fire Up, a beautiful blend of science, technology, and social impact that innovatively solves an age-old problem: wildfire readiness.
""")
    st.link_button("Connect on LinkedIn!", "https://www.linkedin.com/in/anika-jha-24b66723b/")

with col2:
    st.image(ME_PHOTO, use_container_width=True)

st.divider()

#about app
st.markdown("### 🔥 About Fire Up")

col3, col4 = st.columns([0.3,0.7], gap="medium")
with col3:
    st.image(APP_LOGO, use_container_width=True)
with col4:
    st.markdown("""
**Fire Up** is an intuitive yet powerful one-of-a-kind web app that brings together **prediction**, **live detection**, **actionable guidance**, and **real-time evacuation planning** in one cohesive experience, making wildfire readiness more **interactive** and **accessible** to **both** civilians and first responders! 

It dynamically pulls live weather and land data to estimate wildfire **likelihood**, **acres affected**, and **duration** using a custom **LightGBM machine-learning model** trained on multi-source datasets.  

The app also enables users to connect **Raspberry Pi modules** for real-time fire/smoke detection using a **TensorFlow CNN (95% accuracy)**, and includes an interactive **evacuation route planner** powered by **NASA FIRMS** and **Geoapify** APIs. 
                
No existing solution supports live detection with user data. Most also lack dynamic prediction functionality and/or evacuation planning features, and more advanced prediction tools aren't available for the general public.

**Learn more about Fire Up in the video below!** 
                
*Acknowledgements: NASA Earthdata and APIs (FIRMS VIIRS, AppEEARS), CalFire (geojson + shapefiles, past California fire data, qualitative information), other APIs (Open-Meteo, Geoapify), wildfire information sources (USGS, https://www.ready.gov/wildfires, AirNow AQI, NWS Fire Weather, WindTL, Red Cross, NFPA, IBHS, USHS)*
""")

st.divider()

#features
st.markdown("### 🌟 Core Features")
st.markdown("""
<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1.2rem;margin-top:1rem;'>
  <div class='panel' style='text-align:center;'>
    <h4 style='background:var(--accent-grad);-webkit-background-clip:text;color:transparent;'>Prediction</h4>
    <p>Uses ML to forecast wildfire risk with live meteorological and land-cover data.</p>
  </div>
  <div class='panel' style='text-align:center;'>
    <h4 style='background:var(--accent-grad);-webkit-background-clip:text;color:transparent;'>Detection</h4>
    <p>Connects to a Raspberry Pi for live smoke and flame recognition.</p>
  </div>
  <div class='panel' style='text-align:center;'>
    <h4 style='background:var(--accent-grad);-webkit-background-clip:text;color:transparent;'>Evacuation</h4>
    <p>Creates optimized safe routes avoiding active wildfires in real time.</p>
  </div>
  <div class='panel' style='text-align:center;'>
    <h4 style='background:var(--accent-grad);-webkit-background-clip:text;color:transparent;'>Guides</h4>
    <p>Offers concise safety steps and community resources before, during, and after fires.</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

#video
st.markdown("### 🎥 Watch Fire Up in Action")
st.video(VIDEO_URL)

st.divider()

#contact
st.markdown("### 📬 Contact")
st.markdown("Have feedback, ideas, or questions? Use the form below to reach me directly.")

with st.form("contact_form"):
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Your Message", height=120)
    submitted = st.form_submit_button("Send")

if submitted:
    if name and email and message:
        mailto_link = f"mailto:{CONTACT_EMAIL}?subject=Fire%20Up%20Message%20from%20{name}&body={message}%0A%0AReply-to:%20{email}"
        st.markdown(f"[Click here if your email client didn’t open automatically](https://mail.google.com/mail/u/0/?to=anikajhaa8@gmail.com&su=Fire+Up+Message+from+{name}&body={message}&fs=1&tf=cm)")
        st.success("✅ Message ready to send from your email app.")
    else:
        st.warning("Please fill in all fields before submitting.")
