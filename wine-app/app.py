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
