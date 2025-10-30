# viz_fire_duration_holdout.py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error

# ===================== CONFIG =====================
CSV = "ml-model/past_fires_dataset_for_model.csv"
DUR_MODEL_PATH = "ml-model/fire_duration.joblib"      # your saved duration model (pipeline)
LIKE_MODEL_PATH = "ml-model/fire_likeliness.joblib"   # your saved weather→likelihood classifier
TARGET = "DURATION"
CUTOFF_DATE = "2023-09-01"  # train on <= this date, test on > this date

# ===================== LOAD =====================
df = pd.read_csv(CSV)

# Season helper
def month_to_season(m):
    if m in (12, 1, 2): return "WINTER"
    if m in (3, 4, 5):  return "SPRING"
    if m in (6, 7, 8):  return "SUMMER"
    return "FALL"

df["ALARM_DATE"] = pd.to_datetime(df["ALARM_DATE"], errors="coerce")
df["SEASON"] = df["ALARM_DATE"].dt.month.map(month_to_season)

# Drop rows without target/date
df = df[np.isfinite(df[TARGET]) & df["ALARM_DATE"].notna()].copy()

# ===================== FIRE_PROB via likeliness model =====================
clf = joblib.load(LIKE_MODEL_PATH)

df_like = pd.DataFrame(index=df.index)
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

# ===================== FEATURES =====================
# Interactions
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
y = df[TARGET].astype(float).values

# ===================== TIME-BASED SPLIT =====================
# Parse ALARM_DATE as UTC
df["ALARM_DATE"] = pd.to_datetime(df["ALARM_DATE"], utc=True, errors="coerce")

# Make the cutoff UTC-aware, too
cut = pd.to_datetime(CUTOFF_DATE, utc=True)

# Now this works
train_idx = df["ALARM_DATE"] <= cut
test_idx  = df["ALARM_DATE"] >  cut


X_tr, y_tr = X.loc[train_idx], y[train_idx]
X_te, y_te = X.loc[test_idx],  y[test_idx]

print(f"Train rows: {len(X_tr)} | Test rows: {len(X_te)} | Cutoff: {CUTOFF_DATE}")

# If your saved model is a Pipeline, clone it and refit so the test is truly unseen
saved_model = joblib.load(DUR_MODEL_PATH)
model = clone(saved_model)   # same steps/params, fresh weights
model.fit(X_tr, y_tr)

# ===================== EVALUATE ON HOLDOUT =====================
y_pred = model.predict(X_te)

r2  = r2_score(y_te, y_pred)
mae = mean_absolute_error(y_te, y_pred)
print(f"\n=== Holdout Performance (Duration) ===")
print(f"R²: {r2:.3f} | MAE: {mae:.3f} days")

# ===================== PLOTS =====================
# 1) Predicted vs Actual
plt.figure(figsize=(6,6))
plt.scatter(y_te, y_pred, alpha=0.5, edgecolor="k")
diag_min, diag_max = float(np.min([y_te.min(), y_pred.min()])), float(np.max([y_te.max(), y_pred.max()]))
plt.plot([diag_min, diag_max], [diag_min, diag_max], "k--", lw=2)
plt.xlabel("Actual Duration (days)")
plt.ylabel("Predicted Duration (days)")
plt.title(f"Duration: Predicted vs Actual\nR²={r2:.3f}, MAE={mae:.2f} (holdout)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) Residuals histogram
residuals = y_te - y_pred
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=30, edgecolor="k")
plt.axvline(0, color="red", linestyle="--")
plt.title("Residuals (Actual - Predicted) on Holdout")
plt.xlabel("Residuals (days)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

try:
    model_ = model.named_steps['model'].regressor_  # handle TransformedTargetRegressor
    importances = model_.feature_importances_
    features = model.feature_names_in_
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,5))
    plt.barh(np.array(features)[sorted_idx], importances[sorted_idx], color='forestgreen')
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (Duration Model)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Could not plot feature importances:", e)
