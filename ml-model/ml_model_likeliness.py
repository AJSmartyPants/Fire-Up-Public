# ml_model_likeliness.py
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

CSV = "ml-model/CA_Weather_Fire_Dataset_1984-2025.csv"
TARGET = "FIRE_START_DAY"
OUT_MODEL = "ml-model/fire_likeliness.joblib"
OUT_META  = "ml-model/fire_likeliness_meta.json"

df = pd.read_csv(CSV)
df.columns = [c.strip().upper() for c in df.columns]

def month_to_season(m):
    if m in (12, 1, 2):  return "WINTER"
    if m in (3, 4, 5):   return "SPRING"
    if m in (6, 7, 8):   return "SUMMER"
    return "FALL"

df["SEASON"] = df["MONTH"].map(month_to_season)

NUM_FEATURES = [
    "PRECIPITATION", "MAX_TEMP", "MIN_TEMP", "AVG_WIND_SPEED",
    "TEMP_RANGE", "WIND_TEMP_RATIO", "LAGGED_PRECIPITATION", "LAGGED_AVG_WIND_SPEED"
]
CAT_FEATURES = ["SEASON"]

X = df[NUM_FEATURES + CAT_FEATURES]
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", num, NUM_FEATURES), ("cat", cat, CAT_FEATURES)])

candidates = {
    "LogisticRegression": (
        LogisticRegression(max_iter=2000),
        {"model__C": [0.1, 1.0, 3.0]}
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {"model__n_estimators": [200, 500], "model__max_depth": [None, 10]}
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {"model__n_estimators": [100, 300], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 5]}
    ),
    "HistGradientBoosting": (
        HistGradientBoostingClassifier(random_state=42),
        {"model__max_depth": [None, 10], "model__learning_rate": [0.05, 0.1], "model__max_iter": [200, 400]}
    ),
    "SVC": (
        SVC(probability=True, kernel="rbf", random_state=42),
        {"model__C": [1.0, 3.0], "model__gamma": ["scale", 0.1]}
    ),
    "XGBoost": (
        XGBClassifier(
            random_state=42, use_label_encoder=False, eval_metric="logloss",
            tree_method="hist"
        ),
        {"model__n_estimators": [200, 400], "model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]}
    ),
    "LightGBM": (
        LGBMClassifier(random_state=42),
        {"model__n_estimators": [200, 400], "model__learning_rate": [0.05, 0.1], "model__num_leaves": [31, 63]}
    ),
}

leaderboard = []

best_score = -1.0
best_name = None
best_est  = None
best_params = None

for name, (model, grid) in candidates.items():
    pipe = Pipeline([("pre", pre), ("model", model)])
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=5, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    leaderboard.append([name, search.best_score_])
    if search.best_score_ > best_score:
        best_score  = search.best_score_
        best_name   = name
        best_est    = search.best_estimator_
        best_params = search.best_params_

lb = pd.DataFrame(leaderboard, columns=["Model", "CV_ROC_AUC"]).sort_values("CV_ROC_AUC", ascending=False)
print("\n=== MODEL LEADERBOARD (CV ROC AUC) ===")
print(lb.to_string(index=False))

print("\n=== BEST MODEL ===")
print(best_name, best_params)

y_pred  = best_est.predict(X_test)
y_proba = best_est.predict_proba(X_test)[:, 1]

print("\n=== TEST PERFORMANCE ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
print(f"ROC AUC : {roc_auc_score(y_test, y_proba):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

Path("ml-model").mkdir(exist_ok=True)
joblib.dump(best_est, OUT_MODEL)
with open(OUT_META, "w") as f:
    f.write(pd.Series({
        "features": NUM_FEATURES + CAT_FEATURES,
        "best_model": best_name,
        "best_params": best_params
    }).to_json(indent=2))
print(f"\nSaved best model → {OUT_MODEL}")
