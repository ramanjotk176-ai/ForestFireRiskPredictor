# member6_heatmap.py
import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
import joblib
model  = joblib.load("model_xgb.pkl")
scaler = joblib.load("scaler.pkl")
FEATURES = [
    "Temperature","RH","Ws","Rain",
    "FFMC","DMC","DC","ISI","BUI","FWI"
]
# Simulated sensor grid (replace with real GPS + weather data)
np.random.seed(42)
n = 60
sensor_data = pd.DataFrame({
    "lat":         np.random.uniform(36.0, 37.5, n),
    "lon":         np.random.uniform(5.0,  7.5,  n),
    "Temperature": np.random.uniform(25,   42,   n),
    "RH":          np.random.uniform(15,   60,   n),
    "Ws":          np.random.uniform(5,    25,   n),
    "Rain":        np.random.uniform(0,    5,    n),
    "FFMC":        np.random.uniform(70,   96,   n),
    "DMC":         np.random.uniform(10,   120,  n),
    "DC":          np.random.uniform(50,   500,  n),
    "ISI":         np.random.uniform(0,    18,   n),
    "BUI":         np.random.uniform(5,    150,  n),
    "FWI":         np.random.uniform(0,    30,   n),
})
X     = scaler.transform(sensor_data[FEATURES])
probs = model.predict_proba(X)[:, 1]
heat_data = [
    [row["lat"], row["lon"], float(p)]
    for (_, row), p in zip(sensor_data.iterrows(), probs)
]
m = folium.Map(location=[36.7, 6.2], zoom_start=8,
               tiles="CartoDB positron")
HeatMap(
    heat_data, min_opacity=0.3, radius=25, blur=20,
    gradient={"0.3":"blue","0.6":"orange","1.0":"red"}
).add_to(m)
folium.LayerControl().add_to(m)
m.save("fire_risk_heatmap.html")
print("Heatmap saved: fire_risk_heatmap.html")
