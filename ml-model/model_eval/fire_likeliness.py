# viz_fire_likeliness.py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay,
    precision_recall_curve, PrecisionRecallDisplay
)

# === CONFIG ===
CSV = "ml-model/CA_Weather_Fire_Dataset_1984-2025.csv"
MODEL = "ml-model/fire_likeliness.joblib"
TARGET = "FIRE_START_DAY"

# === LOAD ===
df = pd.read_csv(CSV)
df.columns = [c.strip().upper() for c in df.columns]
clf = joblib.load(MODEL)

FEATURES = [
    "PRECIPITATION", "MAX_TEMP", "MIN_TEMP", "AVG_WIND_SPEED",
    "TEMP_RANGE", "WIND_TEMP_RATIO", "LAGGED_PRECIPITATION", "LAGGED_AVG_WIND_SPEED", "SEASON"
]

X = df[FEATURES]
y_true = df[TARGET].astype(int)
y_prob = clf.predict_proba(X)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Fire", "Fire"])
disp.plot(cmap="Reds")
plt.title("Confusion Matrix — Fire Likelihood Model")
plt.show()

# === ROC CURVE ===
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="Fire Likelihood").plot()
plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
plt.show()

# === PRECISION-RECALL CURVE ===
precision, recall, _ = precision_recall_curve(y_true, y_prob)
PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.title("Precision–Recall Curve — Fire Likelihood")
plt.show()
