# member7_app.py  —  run with:  streamlit run member7_app.py
import streamlit as st
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
model  = joblib.load("model_xgb.pkl")
scaler = joblib.load("scaler.pkl")
FEATURES = [
    "Temperature","RH","Ws","Rain",
    "FFMC","DMC","DC","ISI","BUI","FWI"
]
ICONS = {"LOW": "green", "MODERATE": "orange", "HIGH": "red"}
st.set_page_config(
    page_title="Forest Fire Risk Predictor", page_icon="fire"
)
st.title("Forest Fire Risk Predictor")
st.markdown(
    "Adjust environmental conditions below to estimate fire risk."
)
col1, col2 = st.columns(2)
with col1:
    temp = st.slider("Temperature (C)", 10, 50, 30)
    rh   = st.slider("Relative Humidity (%)", 5, 100, 40)
    ws   = st.slider("Wind Speed (km/h)", 0, 40, 15)
    rain = st.slider("Rain (mm)", 0.0, 20.0, 0.0)
    ffmc = st.slider("FFMC", 60.0, 100.0, 85.0)
with col2:
    dmc = st.slider("DMC",  0.0, 200.0,  50.0)
    dc  = st.slider("DC",   0.0, 800.0, 200.0)
    isi = st.slider("ISI",  0.0,  30.0,   8.0)
    bui = st.slider("BUI",  0.0, 250.0,  60.0)
    fwi = st.slider("FWI",  0.0,  60.0,  15.0)
if st.button("Predict Fire Risk"):
    vals   = np.array([[temp,rh,ws,rain,ffmc,dmc,dc,isi,bui,fwi]])
    scaled = scaler.transform(vals)
    prob   = model.predict_proba(scaled)[0][1]
    label  = ("HIGH"     if prob >= 0.70 else
              "MODERATE" if prob >= 0.35 else "LOW")
    st.metric("Fire Probability", f"{prob:.2%}")
    emoji = {"LOW":"green circle","MODERATE":"yellow","HIGH":"red"}
    st.markdown(f"### Risk Level: {label}")
    st.progress(float(prob))
    if label == "HIGH":
        st.error("ALERT: High risk! Notify forest dept immediately.")
    elif label == "MODERATE":
        st.warning("WARNING: Moderate risk. Increase patrols.")
    else:
        st.success("INFO: Low risk. Normal monitoring.")
    # Embedded risk map
    m = folium.Map(location=[36.7, 6.2], zoom_start=7)
    folium.CircleMarker(
        [36.7, 6.2], radius=20,
        color=ICONS[label], fill=True,
        popup=f"Risk: {label} ({prob:.2%})"
    ).add_to(m)
    st_folium(m, width=700, height=350)
