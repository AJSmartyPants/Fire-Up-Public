import streamlit as st
import streamlit_shadcn_ui as ui
import streamlit_js_eval
import numpy as np
import json 
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import time
import joblib
from datetime import datetime
from lib.theme import inject_theme

#control cache
if "refresh_seed" not in st.session_state:
    st.session_state.refresh_seed = 0.0

def _fnum(x, default=math.nan):
    """Safely cast to float; None/''/bad -> default (nan)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

#setup 
GEOJSON_PATH = "data/ca_regions.geojson"
CENTROIDS_CSV = "data/ca_centroids.csv"
LIKELINESS_MODEL_PATH = "models/fire_likeliness.joblib"
ACRES_MODEL_PATH     = "models/fire_acres.joblib"
DURATION_MODEL_PATH  = "models/fire_duration.joblib"

@st.cache_resource(show_spinner=False)
def _load_acres_model():
    return joblib.load(ACRES_MODEL_PATH)

@st.cache_resource(show_spinner=False)
def _load_duration_model():
    return joblib.load(DURATION_MODEL_PATH)

def _predict_generic(df: pd.DataFrame, model):
    """
    Make a prediction with either a regressor or a classifier model.
    - Ensures required columns exist (creates missing with NaN)
    - Orders columns consistently
    - Coerces to numeric
    """
    required = [
        "likelihood", "lat","lon","LC_Type1","LST_Day_1km","NDVI","ET_500m","Percent_Tree_Cover",
        "om_temp_max_c","om_wind_speed_max_ms","om_soil_temp_mean_c",
        "om_rel_humidity_mean","om_vpd_max_kpa","om_soil_moisture_0_7cm_mean",
        "FIRE_PROB","LSTxNDVI","VPDxNDVI","LSTxVPD"
    ]

    X = df.copy()

    #add missing columns as NaN
    for c in required:
        if c not in X.columns:
            X[c] = np.nan

    #order columns exactly as required
    X = X[required]

    #all numbers for model
    X = X.apply(pd.to_numeric, errors="coerce")

    #predict
    if hasattr(model, "predict_proba"):
        y = model.predict_proba(X)[:, 1]
    else:
        y = model.predict(X)

    return pd.Series(pd.to_numeric(y, errors="coerce"), index=X.index, dtype="float64")

def _scale_series(s: pd.Series, out_min: float, out_max: float, q=(0.05, 0.95)) -> pd.Series:
    lo = float(s.quantile(q[0])) if s.notna().any() else 0.0
    hi = float(s.quantile(q[1])) if s.notna().any() else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.full(len(s), (out_min + out_max)/2.0), index=s.index)
    s_clip = s.clip(lo, hi)
    return out_min + (s_clip - lo) * (out_max - out_min) / (hi - lo)

FEATURE_COLS = [
    "PRECIPITATION", "MAX_TEMP", "MIN_TEMP", "AVG_WIND_SPEED",
    "TEMP_RANGE", "WIND_TEMP_RATIO", "LAGGED_PRECIPITATION",
    "LAGGED_AVG_WIND_SPEED", "SEASON"
]

#make season column
def _season_from_month(m: int) -> str:
    if m in (12,1,2):  return "WINTER"
    if m in (3,4,5):   return "SPRING"
    if m in (6,7,8):   return "SUMMER"
    return "FALL"

#cache model once per session
@st.cache_resource(show_spinner=False)
def _load_model():
    return joblib.load(LIKELINESS_MODEL_PATH)

#cache each (lat, lon) fetch 
@st.cache_data(show_spinner=False)
def _fetch_open_meteo(lat: float, lon: float, seed: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    daily_vars = [
        "precipitation_sum",
        "temperature_2m_max",
        "temperature_2m_min",
        "wind_speed_10m_mean",
        "relative_humidity_2m_mean",
        "vapor_pressure_deficit_max",
        "soil_temperature_0_to_7cm_mean",
        "soil_moisture_0_to_7cm_mean",
    ]
    params = {
        "latitude":  lat,
        "longitude": lon,
        "daily": ",".join(daily_vars),
        "timezone": "auto",
        "past_days": 1,
        "forecast_days": 1,
        #seed is only to vary the cache key; it’s unused in the request
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    d = r.json().get("daily", {}) or {}

    def _last(a, idx=-1, default=None):
        try:
            return a[idx]
        except Exception:
            return default

    def _getf(key, idx=-1):
        return _fnum(_last(d.get(key, []) or [], idx, None))

    out = {
        "today": {
            "PRECIPITATION":               _getf("precipitation_sum", -1),
            "MAX_TEMP":                    _getf("temperature_2m_max", -1),
            "MIN_TEMP":                    _getf("temperature_2m_min", -1),
            "AVG_WIND_SPEED":              _getf("wind_speed_10m_mean", -1),
            "om_rel_humidity_mean":        _getf("relative_humidity_2m_mean", -1),
            "om_vpd_max_kpa":              _getf("vapor_pressure_deficit_max", -1),
            "om_soil_temp_mean_c":         _getf("soil_temperature_0_to_7cm_mean", -1),
            "om_soil_moisture_0_7cm_mean": _getf("soil_moisture_0_to_7cm_mean", -1),
            "DATE":                        _last(d.get("time", []) or [], -1, None),
        },
        "yesterday": {
            "PRECIPITATION":   _getf("precipitation_sum", -2),
            "AVG_WIND_SPEED":  _getf("wind_speed_10m_mean", -2),
            "DATE":            _last(d.get("time", []) or [], -2, None),
        }
    }
    return out


def _features_from_open_meteo(lat: float, lon: float, seed: float) -> dict:
    w = _fetch_open_meteo(lat, lon, seed)
    t_max = w["today"]["MAX_TEMP"]
    t_min = w["today"]["MIN_TEMP"]
    wind  = w["today"]["AVG_WIND_SPEED"]

    temp_range = (t_max - t_min) if (pd.notna(t_max) and pd.notna(t_min)) else math.nan
    wind_temp_ratio = (wind / max(t_max, 0.1)) if pd.notna(wind) and pd.notna(t_max) else math.nan

    lag_precip = w["yesterday"]["PRECIPITATION"]
    lag_wind   = w["yesterday"]["AVG_WIND_SPEED"]

    try:
        month = datetime.fromisoformat(w["today"]["DATE"]).month
    except Exception:
        month = datetime.now().month
    season = _season_from_month(month)

    #Base dictionary
    feats = {
        "PRECIPITATION": w["today"]["PRECIPITATION"],
        "MAX_TEMP": t_max,
        "MIN_TEMP": t_min,
        "AVG_WIND_SPEED": wind,
        "TEMP_RANGE": temp_range,
        "WIND_TEMP_RATIO": wind_temp_ratio,
        "LAGGED_PRECIPITATION": lag_precip,
        "LAGGED_AVG_WIND_SPEED": lag_wind,
        "SEASON": season,

        #Open-Meteo “om_*” features 
        "om_temp_max_c": t_max,
        "om_wind_speed_max_ms": wind, 
        "om_soil_temp_mean_c": w["today"]["om_soil_temp_mean_c"],
        "om_rel_humidity_mean": w["today"]["om_rel_humidity_mean"],
        "om_vpd_max_kpa": w["today"]["om_vpd_max_kpa"],
        "om_soil_moisture_0_7cm_mean": w["today"]["om_soil_moisture_0_7cm_mean"],

        #Unfortunately, I was unable to access the ET data because of API limits
        "ET_500m": math.nan,
        "FIRE_PROB": math.nan,
    }

    #Unable to access LST data from AppEEARS realtime as of now
    feats["LST_Day_1km"] = feats["om_soil_temp_mean_c"]

    #Derived combos (will be completed after merging NDVI)
    feats["LSTxNDVI"] = math.nan
    feats["VPDxNDVI"] = math.nan
    feats["LSTxVPD"]  = math.nan

    return feats


def _predict_likeliness(df_features: pd.DataFrame) -> pd.Series:
    model = _load_model()
    if "SEASON" in df_features.columns and not hasattr(model, "transform"):
        season_map = {"WINTER":0, "SPRING":1, "SUMMER":2, "FALL":3}
        df_features = df_features.copy()
        df_features["SEASON"] = df_features["SEASON"].map(season_map).astype("Int64")

    X = df_features[FEATURE_COLS]

    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(X)[:, 1] * 100.0, index=X.index)

    yhat = model.predict(X)
    if pd.api.types.is_float_dtype(yhat) and 0 <= np.nanmin(yhat) <= np.nanmax(yhat) <= 1:
        return pd.Series(yhat * 100.0, index=X.index)

    y = pd.Series(yhat, dtype="float64")
    rng = y.max() - y.min()
    return ((y - y.min()) / rng * 100.0) if rng > 0 else (y * 0.0)

def load_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_centroids(path):
    return pd.read_csv(path)

@st.cache_data(show_spinner=True)
def build_features_for_centroids(centroids_df: pd.DataFrame, seed: float) -> pd.DataFrame:
    #fetch Open-Meteo features for every centroid
    rows = []
    total = len(centroids_df)
    pb = st.progress(0, text="Fetching weather features…")

    for i, row in centroids_df.reset_index(drop=True).iterrows():
        feats = _features_from_open_meteo(float(row["lat"]), float(row["lon"]), seed)
        out = {"region_id": row.get("region_id", None), "lat": row["lat"], "lon": row["lon"]}
        out.update(feats)
        rows.append(out)
        if i % 5 == 0 or i == total - 1:
            pb.progress((i + 1) / total)
        time.sleep(0.01) 

    pb.empty()
    feats_df = pd.DataFrame(rows)

    #predict likelihood (0–100)
    feats_df["likelihood"] = _predict_likeliness(feats_df)

    #merge static AppEARS data
    static = pd.read_csv("data/ca_static_appears.csv")  # must have: region_id, LC_Type1, NDVI, Percent_Tree_Cover
    if "region_id" not in feats_df.columns:
        raise RuntimeError("centroids / features are missing region_id; required for static merge.")
    feats_df = feats_df.merge(
        static[["region_id", "LC_Type1", "NDVI", "Percent_Tree_Cover"]],
        on="region_id",
        how="left",
        validate="m:1"
    )

    #combos 
    feats_df["LST_Day_1km"] = feats_df["om_soil_temp_mean_c"]
    feats_df["LSTxNDVI"]    = feats_df["LST_Day_1km"] * feats_df["NDVI"]
    feats_df["VPDxNDVI"]    = feats_df["om_vpd_max_kpa"] * feats_df["NDVI"]
    feats_df["LSTxVPD"]     = feats_df["LST_Day_1km"] * feats_df["om_vpd_max_kpa"]

    #acres & duration models (use likelihood as 0–1)
    acres_model    = _load_acres_model()
    duration_model = _load_duration_model()
    df_for_downstream = feats_df.copy()
    df_for_downstream["likelihood"] = df_for_downstream["likelihood"] / 100.0

    feats_df["pred_acres"]    = _predict_generic(df_for_downstream, acres_model)
    feats_df["pred_duration"] = _predict_generic(df_for_downstream, duration_model)

    #write latest update for reference
    feats_df.to_csv("data/ca_features_latest.csv", index=False)

    return feats_df

def load_latest_features() -> pd.DataFrame:
    return pd.read_csv("data/ca_features_latest.csv")

def render_ca_prediction_map():
    regions        = load_geojson(GEOJSON_PATH)
    base_centroids = load_centroids(CENTROIDS_CSV)
    seed = st.session_state.get("refresh_seed", 0.0)
    build_features_for_centroids(base_centroids, seed)
    centroids = load_latest_features()

    #compute the user point separately and append
    seed = st.session_state.get("refresh_seed", 0.0)
    user_feats = _features_from_open_meteo(ulat, ulon, seed)
    user_feats["LST_Day_1km"] = user_feats["om_soil_temp_mean_c"]

    #borrow nearest static for the user pin
    static = pd.read_csv("data/ca_static_appears.csv")
    static["dist2"] = (static["lat"]-ulat)**2 + (static["lon"]-ulon)**2
    nearest = static.loc[static["dist2"].idxmin()]
    for c in ["LC_Type1","NDVI","Percent_Tree_Cover"]:
        user_feats[c] = float(nearest.get(c, np.nan))

    user_feats["LSTxNDVI"] = user_feats["LST_Day_1km"] * user_feats["NDVI"]
    user_feats["VPDxNDVI"] = user_feats["om_vpd_max_kpa"] * user_feats["NDVI"]
    user_feats["LSTxVPD"]  = user_feats["LST_Day_1km"] * user_feats["om_vpd_max_kpa"]

    user_like = float(_predict_likeliness(pd.DataFrame([user_feats]))[0])

    acres_model    = _load_acres_model()
    duration_model = _load_duration_model()
    tmp = pd.DataFrame([user_feats]).copy()
    tmp["likelihood"] = user_like / 100.0
    user_acres    = float(_predict_generic(tmp, acres_model)[0])
    user_duration = float(_predict_generic(tmp, duration_model)[0])

    user_row = pd.DataFrame([{**user_feats,
        "region_id": "User",
        "lat": ulat, "lon": ulon,
        "likelihood": user_like,
        "pred_acres": user_acres,
        "pred_duration": user_duration
    }])

    centroids = pd.concat([centroids, user_row], ignore_index=True)

    #VIS
    opacity_heat = st.slider("Heat opacity", 0.0, 1.0, 0.85, 0.05, key="heat_opacity")

    fire_scale = [
        [0.00, "rgb(255,255,178)"],
        [0.25, "rgb(254,204,92)"],
        [0.50, "rgb(253,141,60)"],
        [0.75, "rgb(240,59,32)"],
        [1.00, "rgb(189,0,38)"],
    ]
    fire_scale_alpha = [
        [0.00, "rgba(255,255,178,0.35)"],
        [0.25, "rgba(254,204,92,0.35)"],
        [0.50, "rgba(253,141,60,0.35)"],
        [0.75, "rgba(240,59,32,0.35)"],
        [1.00, "rgba(189,0,38,0.35)"],
    ]

    fig = go.Figure()

    #county fill by relative likelihood 
    #compute per-county centroid row (join via region_id)
    county_like = centroids.dropna(subset=["region_id"]).groupby("region_id", as_index=False)["likelihood"].mean()
    if "User" in county_like["region_id"].values:
        county_like = county_like[county_like["region_id"] != "User"]

    if not county_like.empty:
        # relative scale 1-100 
        mn, mx = float(county_like["likelihood"].min()), float(county_like["likelihood"].max())
        if mx > mn:
            county_like["rel_like"] = 1.0 + 99.0 * (county_like["likelihood"] - mn) / (mx - mn)
        else:
            county_like["rel_like"] = 50.0

        locs, relvals, absvals, cnames = [], [], [], []
        for f in regions["features"]:
            rid = f["properties"].get("region_id")
            nm  = f["properties"].get("region_name") or f["properties"].get("CountyName") or str(rid)
            row = county_like[county_like["region_id"] == rid]
            if not row.empty:
                locs.append(rid)
                relvals.append(float(row["rel_like"].iloc[0]))
                absvals.append(float(row["likelihood"].iloc[0]))
                cnames.append(nm)

        if locs:
            fig.add_trace(go.Choroplethmapbox(
                geojson=regions,
                locations=locs,
                z=relvals,
                zmin=1, zmax=100,
                colorscale=fire_scale_alpha,
                showscale=False,
                featureidkey="properties.region_id",
                name="Relative likelihood (county)",
                customdata=np.stack([cnames, absvals, relvals], axis=-1),
                hovertemplate=("County: %{customdata[0]}<br>"
                               "Likelihood: %{customdata[1]:.1f}%<br>"
                               "Relative Likelihood: %{customdata[2]:.0f}/100"
                               "<extra></extra>")
            ))

    #Heat radius by predicted acres
    #Build quantile bins on acres, radius 25-90 px
    if "pred_acres" in centroids.columns:
        acres_bins = pd.qcut(centroids["pred_acres"].fillna(0.0), q=[0, .2, .4, .6, .8, 1.0], duplicates="drop")
        bin_centers = {}
        for b in acres_bins.cat.categories:
            lo, hi = float(b.left), float(b.right)
            mid = (lo + hi) / 2.0
            radius_px = float(_scale_series(pd.Series([mid]), 25, 90, q=(0,1)).iloc[0])
            bin_centers[str(b)] = radius_px

        #add one heat layer per bin
        for b in acres_bins.cat.categories:
            mask = (acres_bins.astype(str) == str(b))
            sub  = centroids.loc[mask]
            if sub.empty: 
                continue
            fig.add_trace(go.Densitymapbox(
                lat=sub["lat"], lon=sub["lon"], z=sub["likelihood"],
                radius=bin_centers[str(b)],
                colorscale=fire_scale,
                zmin=0, zmax=100,
                opacity=opacity_heat,
                hovertemplate="Likelihood: %{z:.1f}%<extra></extra>",
                showscale=False,
                name="_heat"
            ))
        #single colorbar (separate dummy coloraxis via a tiny hidden trace)
        fig.add_trace(go.Densitymapbox(
            lat=[centroids["lat"].iloc[0]], lon=[centroids["lon"].iloc[0]], z=[centroids["likelihood"].iloc[0]],
            radius=1, colorscale=fire_scale, zmin=0, zmax=100, opacity=0.0, showscale=True,
            colorbar=dict(title="Fire likelihood (%)", ticksuffix="%", tickvals=[0,20,40,60,80,100], x=1.02),
            name="_legend"
        ))

    #centroid dots: size by duration, color by likelihood
    size_by_duration = _scale_series(centroids["pred_duration"].fillna(0.0), 6, 20)
    fig.add_trace(go.Scattermapbox(
        lat=centroids["lat"], lon=centroids["lon"], mode="markers",
        marker=dict(
            size=size_by_duration,
            color=centroids["likelihood"],
            colorscale=fire_scale, cmin=0, cmax=100, opacity=0.95
        ),
        customdata=np.stack([
            centroids.get("region_id", pd.Series([""]*len(centroids))).fillna(""),
            centroids["likelihood"],
            centroids.get("pred_acres", pd.Series([np.nan]*len(centroids))),
            centroids.get("pred_duration", pd.Series([np.nan]*len(centroids))),
        ], axis=-1),
        hovertemplate=("Region: %{customdata[0]}<br>"
                       "Likelihood: %{customdata[1]:.1f}%<br>"
                       "Pred. Acres: %{customdata[2]:,.0f}<br>"
                       "Severity (duration): %{customdata[3]:.1f}<extra></extra>"),
        showlegend=False
    ))

    #thin black outlines on top 
    fig.update_layout(mapbox_layers=[{
        "sourcetype": "geojson",
        "source": regions,
        "type": "line",
        "color": "black",
        "line": {"width": 1},
    }])

    #user pin with full hover 
    fig.add_trace(go.Scattermapbox(
        lat=[ulat], lon=[ulon], mode="markers+text",
        text=["Your Location"], textposition="top center",
        marker=dict(size=12, color="black", symbol="circle"),
        hovertemplate=(f"Your Location<br>"
                       f"Likelihood: {user_like:.1f}%<br>"
                       f"Pred. Acres: {user_acres:,.0f}<br>"
                       f"Severity (duration): {user_duration:.1f}"
                       "<extra></extra>"),
        showlegend=False
    ))

    #interactivity & layout
    fig.update_layout(
        uirevision="keep",
        mapbox_style="open-street-map",
        mapbox_center={"lat": 37.25, "lon": -119.5},
        mapbox_zoom=4.8,
        height=560,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    st.caption("Color = likelihood, radius = predicted acres, dot size = predicted duration. Counties are filled by relative likelihood (this view).")
    st.plotly_chart(fig, use_container_width=True,
        config={"displayModeBar": True, "scrollZoom": True, "doubleClick": "reset", "responsive": True})


st.set_page_config(
    page_title="🔥 Home - Fire Up",
    page_icon="assets/Fire-Up-App-Logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)
inject_theme()

st.subheader("Home")
st.caption("Predictions for a chosen point and your location (please allow location access) • Live fire tracker for CA")
location = streamlit_js_eval.get_geolocation()
ulat = float(location['coords']['latitude'])
ulon = float(location['coords']['longitude'])

left, right = st.columns([1,1])

@st.fragment
def col1():
    st.markdown("""
    <div class="panel">
    <div><strong>Prediction Map - California</strong></div>
    <div class="subtle">Click anywhere in California to request a prediction. Color represents wildfire likeliness, radius represents expected areas affected, and dot size represents intensity (predicted fire duration).</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    render_ca_prediction_map()
    st.write("")
    if st.button("🔄 Refresh predictions"):
        st.session_state.refresh_seed = time.time()  #new seed makes fresh fetch
        st.success("Refreshing predictions…")
        st.rerun()


FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/usfs/api/area/csv/ENTER_YOUR_MAP_KEY/VIIRS_SNPP_NRT/world/2"
@st.cache_data(ttl=3600, show_spinner=True)
def load_firms_data(url):
    df = pd.read_csv(url)
    df["acq_datetime"] = pd.to_datetime(df["acq_date"] + " " + df["acq_time"].astype(str).str.zfill(4),
                                        format="%Y-%m-%d %H%M", errors="coerce", utc=True)
    df = df.rename(columns={
        "latitude": "lat",
        "longitude": "lon",
        "frp": "intensity",
        "scan": "scan_km",
        "track": "track_km"
    })
    df["approx_area_km2"] = df["scan_km"] * df["track_km"]
    return df

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = math.pi / 180
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat/2)**2 +
         np.cos(lat1*p) * np.cos(lat2*p) * np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

def circle_points(lat, lon, radius_km, n=360):
    angles = np.linspace(0, 2*math.pi, n)
    dlat = (radius_km / 111.32) * np.sin(angles)
    dlon = (radius_km / (111.32 * np.cos(np.radians(lat)))) * np.cos(angles)
    return pd.DataFrame({"lat": lat + dlat, "lon": lon + dlon})

@st.fragment
def col2():
    st.markdown("""
    <div class="panel">
    <div><strong>Current Fires</strong></div>
    <div class="subtle">Each point represents a fire. FRP, or Fire Radiative Power, can be used to estimate fire intensity and amount of fuel consumed. Higher FRP = Worse Fire</div>
    </div>
    """, unsafe_allow_html=True)
    df = load_firms_data(FIRMS_URL)
    frp_cap = df["intensity"].quantile(0.97)
    df["intensity_clipped"] = df["intensity"].clip(upper=frp_cap)
    radius_km = st.slider("Radius around you (km)", 10, 1000, 100, step=10)
    filter_radius = st.checkbox("Show only fires within radius", value=False)
    if filter_radius:
        df["distance_km"] = haversine_km(ulat, ulon, df["lat"].to_numpy(), df["lon"].to_numpy())
        df = df[df["distance_km"] <= radius_km]
        st.write(f"Showing {len(df):,} fires within {radius_km} km of your location")
    circle_df = circle_points(ulat, ulon, radius_km)
    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="intensity_clipped",
        size="approx_area_km2",
        hover_data=["acq_date", "acq_time", "satellite", "confidence"],
        color_continuous_scale="Turbo",
        opacity=0.7,
        zoom=4, 
        center=dict(lat=ulat, lon=ulon) 
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="FRP (MW)")
    )

    #User location
    fig.add_trace(go.Scattermapbox(
        lat=[ulat],
        lon=[ulon],
        mode="markers+text",
        marker=dict(size=14, symbol="circle", color="black"),
        text=["You"],
        textposition="top center",
        name="Your Location"
    ))

    #Dotted radius circle
    fig.add_trace(go.Scattermapbox(
        lat=circle_df["lat"],
        lon=circle_df["lon"],
        mode="lines",
        line=dict(width=2, color="black"),
        name=f"{radius_km} km radius"
    ))

    st.plotly_chart(fig, use_container_width=True)
    st.write("")

    cols = st.columns(3)
    with cols[0]:
        ui.metric_card(title="Fires Displayed", content=f"{len(df):,}", key="m1")
    with cols[1]:
        ui.metric_card(title="Median FRP (MW)", content=f"{np.nanmedian(df['intensity']):.2f}", key="m2")
    with cols[2]:
        ui.metric_card(title="Median Area (km²)", content=f"{np.nanmedian(df['approx_area_km2']):.3f}", key="m3")
    
    st.text_input("Phone number for alerts (placeholder)", key="phone_alert")
    ui.button("Enable Fire Alerts (placeholder)", key="enable_alerts")
    st.markdown("</div>", unsafe_allow_html=True)

with left:
    col1()

with right:
    col2()
