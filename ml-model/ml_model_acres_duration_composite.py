import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

CSV = "ml-model/past_fires_dataset_for_model.csv"
LIKELI_MODEL = "ml-model/fire_likeliness.joblib"
OUT_DIR = Path("ml-model")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV)

def month_to_season(m):
    if m in (12, 1, 2): return "WINTER"
    if m in (3, 4, 5):  return "SPRING"
    if m in (6, 7, 8):  return "SUMMER"
    return "FALL"
df["SEASON"] = pd.to_datetime(df["ALARM_DATE"], errors="coerce").dt.month.map(month_to_season)

clf = joblib.load(LIKELI_MODEL)

#Recreate the weather feature frame the classifier expects
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

df["LSTxNDVI"] = df["LST_Day_1km"] * df["NDVI"]
df["VPDxNDVI"] = df["om_vpd_max_kpa"] * df["NDVI"]
df["LSTxVPD"]  = df["LST_Day_1km"] * df["om_vpd_max_kpa"]

FEATURES = [
    "lat","lon","LC_Type1","LST_Day_1km","NDVI","ET_500m","Percent_Tree_Cover",
    "om_temp_max_c","om_wind_speed_max_ms","om_soil_temp_mean_c",
    "om_rel_humidity_mean","om_vpd_max_kpa","om_soil_moisture_0_7cm_mean",
    "FIRE_PROB","LSTxNDVI","VPDxNDVI","LSTxVPD"
]

ok = np.isfinite(df[["DURATION", "GIS_ACRES"]]).all(axis=1)
df = df.loc[ok].copy()
X = df[FEATURES].astype(float)
y_dur = df["DURATION"].astype(float)
y_acr = df["GIS_ACRES"].astype(float)

idx = np.arange(len(X))
idx_tr, idx_te = train_test_split(idx, test_size=0.25, random_state=42, shuffle=True)

X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
y_dur_tr, y_dur_te = y_dur.iloc[idx_tr], y_dur.iloc[idx_te]
y_acr_tr, y_acr_te = y_acr.iloc[idx_tr], y_acr.iloc[idx_te]

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), FEATURES)
])

models = {
    #Linear baselines
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=10.0),
    "Lasso": Lasso(alpha=0.001, max_iter=20000),

    #Trees/ensembles (tuned)
    "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=1000, max_features="sqrt", min_samples_leaf=2, n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=42
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=500, max_leaf_nodes=63, l2_regularization=0.1, random_state=42
    ),
    "KNN": KNeighborsRegressor(n_neighbors=11, weights="distance"),
    "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),

    #Boosters
    "XGBoost": XGBRegressor(
        n_estimators=800, max_depth=5, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, min_child_weight=1, gamma=0.1,
        objective="reg:squarederror", tree_method="hist", random_state=42, n_jobs=-1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=1000, learning_rate=0.03,
        num_leaves=63, max_depth=-1,
        min_child_samples=20, reg_lambda=0.5, reg_alpha=0.1, random_state=42
    ),
}

def evaluate(y_true, y_pred):
    return r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred)

def leaderboard(X_tr, y_tr, X_te, y_te, label):
    rows = []
    for name, model in models.items():
        ttr = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
        pipe = Pipeline([("pre", pre), ("model", ttr)])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        r2, mae = evaluate(y_te, preds)
        rows.append([name, r2, mae])
    out = pd.DataFrame(rows, columns=["Model", "R2", "MAE"]).sort_values("MAE")
    print(f"\n=== {label} ===")
    print(out.to_string(index=False))
    return out

res_dur = leaderboard(X_tr, y_dur_tr, X_te, y_dur_te, "DURATION (days)")
res_acr = leaderboard(X_tr, y_acr_tr, X_te, y_acr_te, "GIS_ACRES (acres)")

#export best lightgbm model

#Retrain LightGBM on all data (since leaderboard used a split)
final_lgbm = LGBMRegressor(
    n_estimators=1000, learning_rate=0.03,
        num_leaves=63, max_depth=-1,
        min_child_samples=20, reg_lambda=0.5, reg_alpha=0.1, random_state=42
)

#Build full pipeline (same preprocessing)
from sklearn.compose import TransformedTargetRegressor

#Train for DURATION
ttr_dur = TransformedTargetRegressor(regressor=final_lgbm, func=np.log1p, inverse_func=np.expm1)
pipe_dur = Pipeline([("pre", pre), ("model", ttr_dur)])
pipe_dur.fit(X, y_dur)

#Train for GIS_ACRES
ttr_acr = TransformedTargetRegressor(regressor=final_lgbm, func=np.log1p, inverse_func=np.expm1)
pipe_acr = Pipeline([("pre", pre), ("model", ttr_acr)])
pipe_acr.fit(X, y_acr)

#Save both pipelines
import joblib
joblib.dump(pipe_dur, OUT_DIR / "fire_duration.joblib")
joblib.dump(pipe_acr, OUT_DIR / "fire_acres.joblib")

print("\nExported final LightGBM models:")
