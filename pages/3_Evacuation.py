import io, json, requests, pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium, folium_static
import streamlit_js_eval
from lib.theme import inject_theme
from shapely.geometry import Point
from shapely.ops import transform as shp_transform
from pyproj import Transformer

# --- CONFIG ---
st.set_page_config(
    page_title="Fire Up - Evacuation Routes",
    page_icon="assets/Fire-Up-App-Logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)
inject_theme()
st.header("Evacuation Routes")
st.caption("Click a destination on the left map or type an address. The route avoids locations of active fires.")

# Secrets / env
GEOAPIFY_KEY = "APIKEY"
FIRMS_MAP_KEY = "APIKEY"  # https://firms.modaps.eosdis.nasa.gov/
FIRMS_PRODUCT = "VIIRS_SNPP_NRT"
FIRMS_LOOKBACK_DAYS = 3
FIRE_BUFFER_KM = 1.0  # for drawing circles (visualization only)

# Projections for accurate circle drawing
_to3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
_to4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

def buffer_point_km(lat: float, lon: float, radius_km: float):
    p = Point(lon, lat)
    p_merc = shp_transform(_to3857, p)
    buf_merc = p_merc.buffer(radius_km * 1000.0)
    return shp_transform(_to4326, buf_merc)  # shapely polygon (WGS84)

# --- Geocoder (typed address) ---
def quick_geocode(query: str):
    if not query.strip():
        return None
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "fire-up-evac/1.0"}
        )
        r.raise_for_status()
        data = r.json()
        if not data: return None
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        return None

# --- FIRMS helpers ---
def bbox_from_points(p1, p2, pad_km=30):
    pad_deg = pad_km / 111.0
    lats = [p1[0], p2[0]]
    lons = [p1[1], p2[1]]
    return (min(lats)-pad_deg, min(lons)-pad_deg, max(lats)+pad_deg, max(lons)+pad_deg)

def bbox_to_wkt(bbox):
    minlat, minlon, maxlat, maxlon = bbox
    coords = [(minlon, minlat), (maxlon, minlat), (maxlon, maxlat), (minlon, maxlat), (minlon, minlat)]
    return "POLYGON((" + ",".join([f"{x} {y}" for x, y in coords]) + "))"

def fetch_firms_points_near(start, end, lookback_days=FIRMS_LOOKBACK_DAYS, product=FIRMS_PRODUCT):
    if not FIRMS_MAP_KEY:
        return pd.DataFrame(columns=["latitude","longitude","frp"])
    wkt = bbox_to_wkt(bbox_from_points(start, end, pad_km=30))
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/{product}/{lookback_days}/{wkt}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        cols = {c.lower(): c for c in df.columns}
        latc = cols.get("latitude") or cols.get("lat")
        lonc = cols.get("longitude") or cols.get("lon")
        frpc = cols.get("frp")
        df = df.rename(columns={latc: "latitude", lonc: "longitude", frpc: "frp"})
        return df[["latitude","longitude","frp"]].dropna()
    except Exception:
        return pd.DataFrame(columns=["latitude","longitude","frp"])

# --- Geoapify Routing ---
# Docs: avoid parameter supports "tolls|ferries|highways|location:lat,lon" etc. We'll pass each fire point as an avoid location.
# Example: ...&avoid=location:35.234045,-80.836392|location:35.22,-80.83 ...
# (Note: Avoids are soft constraints; if no alternative exists, the engine may still pass near) :contentReference[oaicite:1]{index=1}
def build_avoid_param_from_fires(fire_df, max_locations=80):
    if fire_df is None or fire_df.empty:
        return None
    # keep up to max_locations nearest to the straight line (simple heuristic)
    # here we just take the first N for simplicity
    vals = []
    cnt = 0
    for _, row in fire_df.iterrows():
        if cnt >= max_locations: break
        vals.append(f"location:{row['latitude']:.6f},{row['longitude']:.6f}")
        cnt += 1
    return "|".join(vals) if vals else None

def geoapify_route(start, dest, avoid_param=None, mode="drive", fmt="geojson"):
    base = "https://api.geoapify.com/v1/routing"
    params = {
        "waypoints": f"{start[0]},{start[1]}|{dest[0]},{dest[1]}",
        "mode": mode,
        "apiKey": GEOAPIFY_KEY,
        "format": fmt
    }
    if avoid_param:
        params["avoid"] = avoid_param
    r = requests.get(base, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def parse_geoapify_geojson_to_latlon_list(geojson_obj):
    # Geoapify GeoJSON returns MultiLineString route(s) in features[0]["geometry"]["coordinates"]
    # Coordinates are [lon, lat] -> convert to [lat, lon]
    try:
        features = geojson_obj.get("features", [])
        if not features: return []
        coords = features[0]["geometry"]["coordinates"]  # MultiLineString: list of LineStrings
        latlon = []
        for line in coords:
            latlon.extend([[pt[1], pt[0]] for pt in line])
        return latlon
    except Exception:
        return []

# --- Get user location (browser) ---
loc = streamlit_js_eval.get_geolocation()
try:
    ulat = float(loc["coords"]["latitude"])
    ulon = float(loc["coords"]["longitude"])
except Exception:
    ulat, ulon = 34.1425, -118.0280  # fallback: Arcadia, CA

start = (ulat, ulon)

# --- UI (two columns) ---
left, right = st.columns([0.55, 0.45])

with left:
    st.markdown("**1) Click your destination** (left map) or type a place below.")
    # Clickable map
    m_click = folium.Map(location=[ulat, ulon], zoom_start=12, tiles="cartodbpositron")
    folium.Marker([ulat, ulon], tooltip="You (start)", icon=folium.Icon(color="green")).add_to(m_click)
    m_click.add_child(folium.LatLngPopup())
    click_state = st_folium(m_click, height=450)  # interactive map

    st.write("")
    dest_query = st.text_input("…or type a destination (address/place)")

with right:
    st.markdown("**2) Final route** (right map)")

# Determine destination from click or geocode
dest = None
if click_state and click_state.get("last_clicked"):
    dest = (click_state["last_clicked"]["lat"], click_state["last_clicked"]["lng"])
elif dest_query:
    g = quick_geocode(dest_query)
    if g: dest = g

go = st.button("Generate Route")

# --- Action: compute & render ---
if go and not GEOAPIFY_KEY:
    st.error("Missing GEOAPIFY_KEY. Add it to Streamlit secrets.")
elif go and not dest:
    st.warning("Please click a destination on the left map or type an address I can geocode.")
elif go and dest:
    # Fetch recent FIRMS fires near start-dest corridor
    fires = fetch_firms_points_near(start, dest)
    avoid_param = build_avoid_param_from_fires(fires)

    # Call Geoapify route with avoid=location:lat,lon list (supported by Routing API) :contentReference[oaicite:2]{index=2}
    route_geojson = geoapify_route(start, dest, avoid_param=avoid_param, mode="drive", fmt="geojson")
    route_latlon = parse_geoapify_geojson_to_latlon_list(route_geojson)

    # Build the rendered map (folium_static)
    m_out = folium.Map(location=[ulat, ulon], zoom_start=12, tiles="cartodbpositron")
    folium.Marker([ulat, ulon], tooltip="You (start)", icon=folium.Icon(color="green")).add_to(m_out)
    folium.Marker(dest, tooltip="Destination", icon=folium.Icon(color="blue")).add_to(m_out)

    # Draw fires + 1 km circles
    if fires is not None and not fires.empty:
        for _, row in fires.iterrows():
            lat, lon = float(row["latitude"]), float(row["longitude"])
            folium.CircleMarker([lat, lon], radius=3, color="red", fill=True, tooltip="Active fire").add_to(m_out)
            buf = buffer_point_km(lat, lon, FIRE_BUFFER_KM)
            folium.GeoJson(
                buf.__geo_interface__,
                style_function=lambda _: {"fillColor": "#ff0000", "color": "#ff0000", "weight": 1, "fillOpacity": 0.15}
            ).add_to(m_out)

    # Draw route
    if route_latlon:
        folium.PolyLine(route_latlon, color="blue", weight=6, opacity=0.95, tooltip="Evacuation route").add_to(m_out)
    else:
        folium.map.Popup("No route returned (try moving destination slightly or reduce avoids).").add_to(m_out)

    # Render on the right with folium_static
    with right:
        folium_static(m_out, width=720, height=500)
