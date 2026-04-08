# member2_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
df = pd.read_csv("algerian_forest_fires.csv", skiprows=1)
df.columns = df.columns.str.strip()
df = df[df["Classes"].notna()].copy()
df["Classes"] = df["Classes"].str.strip()
features = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]
df[features] = df[features].apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)
le = LabelEncoder()
df["target"] = le.fit_transform(df["Classes"])  # fire=1, not fire=0
print("Classes:", le.classes_)
X = df[features]
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le,     "label_encoder.pkl")
np.save("X_train.npy", X_train)
np.save("X_test.npy",  X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy",  y_test)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print("Preprocessing complete. Files saved.")
