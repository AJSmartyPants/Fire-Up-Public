# viz_fire_acres.py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# === CONFIG ===
CSV = "ml-model/past_fires_dataset_for_model.csv"
MODEL = "ml-model/fire_acres.joblib"
LIKELI_MODEL = "ml-model/fire_likeliness.joblib"
TARGET = "GIS_ACRES"

# === LOAD ===
df = pd.read_csv(CSV)
df = df[np.isfinite(df[TARGET])]
model = joblib.load(MODEL)

# === SEASON FEATURE ===
def month_to_season(m):
    if m in (12, 1, 2): return "WINTER"
    if m in (3, 4, 5):  return "SPRING"
    if m in (6, 7, 8):  return "SUMMER"
    return "FALL"
df["SEASON"] = pd.to_datetime(df["ALARM_DATE"], errors="coerce").dt.month.map(month_to_season)

# === FIRE_LIKELINESS via your classifier ===
clf = joblib.load(LIKELI_MODEL)

# Recreate the weather feature frame the classifier expects
df_like = pd.DataFrame()
df_like["MAX_TEMP"] = df["om_temp_max_c"]
df_like["MIN_TEMP"] = df["om_soil_temp_mean_c"] - 5
df_like["AVG_WIND_SPEED"] = df["om_wind_speed_max_ms"]
df_like["PRECIPITATION"] = 0.0
df_like["TEMP_RANGE"] = df_like["MAX_TEMP"] - df_like["MIN_TEMP"]
df_like["WIND_TEMP_RATIO"] = df_like["AVG_WIND_SPEED"] / (df_like["MAX_TEMP"] + 1e-6)
df_like["LAGGED_PRECIPITATION"] = df_like["PRECIPITATION"]
df_like["LAGGED_AVG_WIND_SPEED"] = df_like["AVG_WIND_SPEED"]
df_like["SEASON"] = df["SEASON"]

df["FIRE_PROB"] = clf.predict_proba(df_like)[:, 1]

# === FEATURES & INTERACTIONS ===
df["LSTxNDVI"] = df["LST_Day_1km"] * df["NDVI"]
df["VPDxNDVI"] = df["om_vpd_max_kpa"] * df["NDVI"]
df["LSTxVPD"]  = df["LST_Day_1km"] * df["om_vpd_max_kpa"]

FEATURES = [
    "lat","lon","LC_Type1","LST_Day_1km","NDVI","ET_500m","Percent_Tree_Cover",
    "om_temp_max_c","om_wind_speed_max_ms","om_soil_temp_mean_c",
    "om_rel_humidity_mean","om_vpd_max_kpa","om_soil_moisture_0_7cm_mean",
    "FIRE_PROB","LSTxNDVI","VPDxNDVI","LSTxVPD"
]

X = df[FEATURES]
y_true = df[TARGET].values
y_pred = model.predict(X)

# === METRICS ===
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
print(f"R²: {r2:.3f} | MAE: {mae:.3f} acres")

# === PREDICTED VS ACTUAL SCATTER ===
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.5, color="green", edgecolor="k")
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], "k--", lw=2)
plt.xlabel("Actual Burn Area (acres)")
plt.ylabel("Predicted Burn Area (acres)")
plt.title(f"Predicted vs Actual Burn Area\nR²={r2:.3f}, MAE={mae:.2f}")
plt.grid(True)
plt.show()

# === LOG-SCALE COMPARISON (optional, for large range of acres) ===
plt.figure(figsize=(6,6))
plt.scatter(np.log1p(y_true), np.log1p(y_pred), alpha=0.5, color="seagreen", edgecolor="k")
plt.plot([0, max(np.log1p(y_true))], [0, max(np.log1p(y_true))], "k--", lw=2)
plt.xlabel("Log(Actual Acres + 1)")
plt.ylabel("Log(Predicted Acres + 1)")
plt.title("Predicted vs Actual (Log Scale)")
plt.grid(True)
plt.show()

try:
    model_ = model.named_steps['model'].regressor_  # handle TransformedTargetRegressor
    importances = model_.feature_importances_
    features = model.feature_names_in_
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,5))
    plt.barh(np.array(features)[sorted_idx], importances[sorted_idx], color='forestgreen')
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (For Acres)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Could not plot feature importances:", e)
