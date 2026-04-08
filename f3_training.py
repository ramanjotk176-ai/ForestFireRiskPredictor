# member3_training.py
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")
# Random Forest
rf = RandomForestClassifier(
    n_estimators=100, max_depth=8, random_state=42
)
rf.fit(X_train, y_train)
rf_cv = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1")
print(f"RF Cross-val F1: {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}")
# XGBoost
xgb = XGBClassifier(
    n_estimators=150, max_depth=6, learning_rate=0.1,
    use_label_encoder=False, eval_metric="logloss", random_state=42
)
xgb.fit(X_train, y_train)
xgb_cv = cross_val_score(xgb, X_train, y_train, cv=5, scoring="f1")
print(f"XGB Cross-val F1: {xgb_cv.mean():.4f} +/- {xgb_cv.std():.4f}")
joblib.dump(rf,  "model_rf.pkl")
joblib.dump(xgb, "model_xgb.pkl")
print("Models saved: model_rf.pkl | model_xgb.pkl")
