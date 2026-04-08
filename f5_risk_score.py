# member5_risk_score.py
import numpy as np
import joblib
model  = joblib.load("model_xgb.pkl")
scaler = joblib.load("scaler.pkl")
FEATURES = [
    "Temperature","RH","Ws","Rain",
    "FFMC","DMC","DC","ISI","BUI","FWI"
]
def get_risk_label(prob: float) -> str:
    if prob < 0.35:
        return "LOW"
    elif prob < 0.70:
        return "MODERATE"
    else:
        return "HIGH"
def predict_risk(input_dict: dict) -> dict:
    values = np.array([[input_dict[f] for f in FEATURES]])
    scaled = scaler.transform(values)
    prob   = model.predict_proba(scaled)[0][1]
    label  = get_risk_label(prob)
    messages = {
        "HIGH":     "ALERT: High fire risk! Notify forest dept immediately.",
        "MODERATE": "WARNING: Moderate risk. Increase monitoring frequency.",
        "LOW":      "INFO: Low risk. Routine monitoring advised."
    }
    return {
        "probability": round(prob, 4),
        "risk_level":  label,
        "alert":       messages[label]
    }
# Sample prediction
sample = {
    "Temperature": 34, "RH": 25,  "Ws": 18, "Rain": 0,
    "FFMC": 91.2,  "DMC": 85.5, "DC": 420.3,
    "ISI": 10.1,   "BUI": 95.4, "FWI": 21.6
}
result = predict_risk(sample)
print(f"Probability : {result['probability']}")
print(f"Risk Level  : {result['risk_level']}")
print(f"Alert       : {result['alert']}")
