import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE = Path(__file__).parent

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

@st.cache_resource
def load():
    return (joblib.load(BASE / "wine_model.pkl"),
            joblib.load(BASE / "wine_scaler.pkl"),
            joblib.load(BASE / "wine_stats.pkl"))

model, scaler, stats = load()
cols = stats["columns"]

st.title("🍷 Wine Quality Predictor")
st.write("Enter the chemical properties of a wine to predict if it's likely **good** (quality ≥ 6).")

values = {}
left, right = st.columns(2)
for i, c in enumerate(cols):
    lo, hi, med = stats["min"][c], stats["max"][c], stats["median"][c]
    step = (hi - lo) / 200
    with (left if i % 2 == 0 else right):
        values[c] = st.slider(c, float(lo), float(hi), float(med), float(step))

if st.button("Predict", type="primary"):
    row = pd.DataFrame([values], columns=cols)
    scaled = scaler.transform(row)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    if pred == 1:
        st.success(f"Likely a GOOD wine ({prob:.0%} confidence)")
    else:
        st.error(f"Likely an average/poor wine ({1 - prob:.0%} confidence)")
    st.progress(float(prob), text=f"P(good) = {prob:.2f}")
