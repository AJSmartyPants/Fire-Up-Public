# **Fire Up is published!**
### [View the live app here](https://fire-up.streamlit.app) (https://fire-up.streamlit.app)

---

## 🔥 About Fire Up by Anika Jha

**Fire Up**, created by Anika Jha, is an intelligent, interactive wildfire preparedness web app that combines **prediction**, **live detection**, **evacuation route planning**, and **safety guidance** in one unified, user-friendly experience.

The app uses real-time weather and land-cover data to predict **wildfire likelihood, acres affected, and duration (severity)**. It’s designed to empower both civilians and first responders with smarter tools for readiness and response.

> 🎥 **Watch the demo video:**  
> [![Watch on YouTube](https://img.shields.io/badge/▶️%20Watch%20on-YouTube-red)](https://youtu.be/M6boFqd9h_E)
(https://youtu.be/M6boFqd9h_E)

---

## Key Features

### 🧠 **Dynamic Wildfire Prediction**
- Uses live inputs from **Open-Meteo API**, **NASA AppEEARS**, and **CalFire GIS** data.  
- Processes data on land cover, vegetation index (NDVI), soil moisture, and meteorological variables.  
- Employs a custom **LightGBM machine-learning model** to estimate:
  - Wildfire **likelihood**
  - **Acres potentially affected**
  - **Fire duration / severity**
- Map visualization built with **Plotly Mapbox** — counties shaded by risk, marker size scales with predicted duration, and radius scales with acres affected.

### 📡 **Live Fire & Smoke Detection**
- Integrates with **Raspberry Pi 5 + sensor modules** (temperature, humidity, light, camera).  
- Runs a **TensorFlow Keras CNN** trained to 95 % accuracy for detecting **fire / smoke / safe** frames.  
- Users can paste their Flask API URL directly into the app to stream live detection results from their own devices.

### 🧭 **Evacuation Route Planner**
- Generates real-time optimized escape routes with the **Geoapify Routing API**.  
- Automatically avoids active wildfire zones using **NASA FIRMS** satellite hotspot data.  
- Interactive map built with **Folium**, allowing users to visualize safe paths dynamically.

### 📚 **Fire Guide + SOS System**
- A compact fire safety and preparedness guide written in plain language for quick reference.  
- The **SOS page** lets users:
  - Share live location and status  
  - Select from pre-written emergency messages  
  - Instantly open 911 and contact local community resources  
  - Send status updates via email or SMS (coming soon with Twilio)

### ℹ️ **About Page**
- Personal developer story and mission statement  
- Embedded video presentation  
- Contact form that launches a pre-filled email directly to the developer

---

## ⚙️ Tech Stack

| Category | Technologies |
|-----------|---------------|
| **Frontend / UI** | Streamlit • ShadCN UI • HTML/CSS custom theme |
| **Backend & APIs** | Python • Requests • Geoapify API • Open-Meteo API • NASA AppEEARS |
| **Machine Learning** | LightGBM • Scikit-learn • TensorFlow Keras CNN |
| **Mapping & GIS** | Plotly Express • Folium • GeoJSON • Pandas/GeoPandas |
| **Hardware Integration** | Raspberry Pi 5 • Flask API • OpenCV for live video |
| **Deployment** | Streamlit Cloud (App link above) |

---

## 🧪 Try It On Your Own

You can clone and run **Fire Up** locally:

git clone https://github.com/AJSmartyPants/Fire-Up-Public.git
cd Fire-Up-Public
pip install -r requirements.txt
streamlit run Welcome.py

### ⚠️ Important Notes

- 🌤 **API Keys Required**  
  You must create your own **Geoapify** and **Open-Meteo** API keys (both have free tiers).  
  Add them directly in the script or configure them as environment variables if deploying to Streamlit Cloud.

- 📂 **Large Datasets Excluded**  
  Due to GitHub file size limits, the large **NASA AppEEARS** and **CalFire** datasets used for model training are **not included** in this repository.  
  To reproduce the training data:
  - Use the script [`scripts/fetch_static_appeears.py`](scripts/fetch_static_appeears.py)  
  - Fetch your own AppEEARS layers (NDVI, Tree Cover, Land Cover, etc.)  
  - Merge with your regional centroid CSVs before running model training.

- 🧠 **Model Training**  
  The pre-trained `.joblib` models (`fire_likeliness.joblib`, `fire_acres.joblib`, and `fire_duration.joblib`) are provided for demonstration.  
  If you wish to retrain them:
  - Update your datasets under `data/`  
  - Run the provided ML scripts or notebooks  
  - Replace the existing model files under `models/`

## 🍓 Raspberry Pi Integration (Live Detection Module)

The Raspberry Pi module allows you to stream real-time environmental and camera data directly into the **Fire Up** app.
It runs a local Flask server that the web app connects to in order to display live sensor data and video.

The file used is located at:

raspberry-pi-code/fire-up-raspberry-pi-code.py

This script captures:

* **Temperature & Humidity** from a DHT20 sensor
* **Light intensity (lux)** from a BH1750 sensor
* **Live video feed** from the Raspberry Pi camera
* And exposes them via a Flask API that connects to Fire Up’s Detections page

---

## 🔧 Required Hardware

To use the Raspberry Pi functionality, you’ll need:

* Raspberry Pi 4 (recommended)
* Raspberry Pi OS (Bullseye or Bookworm)
* DFRobot DHT20 (Temperature & Humidity) — I²C address `0x38`
* BH1750 Light Sensor — I²C address `0x5C`
* Raspberry Pi Camera Module (or compatible USB camera)
* Stable Wi-Fi or Ethernet connection

---

## ⚙️ Raspberry Pi Setup Instructions

### 1. Update your Pi and enable interfaces

Open a terminal on the Pi and run:

```bash
sudo apt update && sudo apt upgrade -y
sudo raspi-config
```

Then go to:

* Interface Options → **Enable I2C**
* Interface Options → **Enable Camera**

Reboot your Pi:

```bash
sudo reboot
```

---

### 2. Install required packages

In the terminal, install these dependencies:

```bash
sudo apt install -y python3-pip python3-opencv python3-picamera2 python3-smbus i2c-tools
pip3 install flask DFRobot_DHT20
```

---

### 3. Confirm sensors are connected correctly

Run:

```bash
i2cdetect -y 1
```

You should see:

* `38` → DHT20 sensor
* `5c` → BH1750 light sensor

If not, check your wiring.

---

### 4. Run the Raspberry Pi script

Navigate to the folder and run the script:

```bash
cd raspberry-pi-code
python3 fire-up-raspberry-pi-code.py
```

You should see output similar to:

```
Running on http://0.0.0.0:8000
```

This means the local Flask server is active.

Available endpoints:

* `/mjpeg` → live camera stream
* `/frame.jpg` → single image frame
* `/sensors` → live temperature, humidity, and light level (JSON)

---

## 🔗 Connecting to the Fire Up App

1. Find your Raspberry Pi’s IP address:

```bash
hostname -I
```

2. In **Fire Up → Detections Page**, paste the URL:

```text
http://<your-pi-ip>:8000
```

Example:

```text
http://192.168.1.14:8000
```

Once connected, the app will display:

* Live camera feed
* Real-time environmental data
* Fire/smoke classification using the AI model

---

## 📌 Notes

* The temperature includes a −10°C adjustment to account for Pi overheating. You can change this in the script as needed.
* Use a proper **5V 3A power supply** for stability.
* Place the system in a well-ventilated, outdoor-protected space for best results.
* This setup can be expanded to send alerts using services like Twilio or email APIs.

Your Raspberry Pi now acts as a **real-time wildfire detection node** chained directly into **Fire Up**.


---

## 🧰 Repository Structure  
Fire-Up-Public/  
│  
├── pages/             # Streamlit app pages (Home, Detections, Evacuation, etc.)  
├── models/            # Pre-trained LightGBM models (.joblib) and CNN for fire/smoke classification in live feed  
├── data/              # Processed static + live datasets  
├── scripts/           # Helper scripts (AppEEARS fetch, preprocessing, etc.)  
├── lib/  
│   └── theme.py       # Global CSS theme for consistent UI styling  
├── assets/            # App logo and media files  
├── raspberry-pi-code/  
│   └── fire-up-raspberry-pi-code.py   # Code for Raspberry Pi module (runs Flask server for live monitoring)  
├── Welcome.py         # App entry point (landing page)  
├── requirements.txt   # Dependencies for Streamlit deployment  
└── ml-model/          # Scripts & results used for model training and evaluation  

## 💬 Feedback & Contact

Got feedback, questions, or ideas for collaboration?  
📧 **anikajhaa8@gmail.com**

> Together, we can use data, innovation, and empathy to fight wildfires before they spread. 
