import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import timedelta
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GreenGrid | Renewable Forecast",
    page_icon="⚡",
    layout="wide"
)

# ---------------- DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071a18 0%, #102b25 55%, #071512 100%);
    color: #edf7f1;
}
h1, h2, h3, p, label { color: #edf7f1 !important; }
.hero {
    padding: 28px 32px;
    border: 1px solid #245b49;
    border-radius: 22px;
    background: linear-gradient(110deg, #123b30, #10251f);
    margin-bottom: 22px;
}
.hero small { color: #9dd9b8; letter-spacing: 2px; }
.hero h1 { font-size: 42px; margin: 8px 0; }
.hero p { color: #b9d4c7 !important; font-size: 16px; }
.metric-card {
    background: #15352d;
    border: 1px solid #2d6652;
    border-radius: 18px;
    padding: 22px;
    min-height: 140px;
}
.metric-label { color: #a9cbb9; font-size: 14px; }
.metric-value { color: #71e0a2; font-size: 32px; font-weight: 700; }
.metric-note { color: #b9d4c7; font-size: 12px; }
div.stButton > button {
    background: linear-gradient(90deg, #31b875, #65d99b);
    color: #062016;
    border: 0;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: 700;
    width: 100%;
}
div.stButton > button:hover {
    background: #8af0b5;
    color: #062016;
}
[data-testid="stMetric"] {
    background: #15352d;
    padding: 18px;
    border: 1px solid #2d6652;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FILE LOCATIONS ----------------
BASE = Path(__file__).parent

MODEL_PATH = BASE / "renewable_gru_model_FINAL_1H.keras"
FEATURE_SCALER_PATH = BASE / "feature_scaler_FINAL_1H.pkl"
TARGET_SCALER_PATH = BASE / "target_scaler_FINAL_1H.pkl"

# Find the project dataset
possible_csvs = [
    BASE / "renewable_energy_project_dataset.csv",
    BASE / "reg_forecasting_v2_project_clean.csv",
]

DATA_PATH = next((p for p in possible_csvs if p.exists()), None)

if DATA_PATH is None:
    for p in BASE.glob("*.csv"):
        try:
            cols = pd.read_csv(p, nrows=2).columns
            if "datetime" in cols and "Total_Renewable" in cols:
                DATA_PATH = p
                break
        except Exception:
            pass

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df = df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    if "sin_time" not in df.columns:
        hour = df["datetime"].dt.hour
        df["sin_time"] = np.sin(2 * np.pi * hour / 24)
    if "cos_time" not in df.columns:
        hour = df["datetime"].dt.hour
        df["cos_time"] = np.cos(2 * np.pi * hour / 24)
    if "sin_doy" not in df.columns:
        doy = df["datetime"].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    if "cos_doy" not in df.columns:
        doy = df["datetime"].dt.dayofyear
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    return df

@st.cache_resource
def load_model_and_scalers():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(FEATURE_SCALER_PATH, "rb") as f:
        feature_scaler = pickle.load(f)
    with open(TARGET_SCALER_PATH, "rb") as f:
        target_scaler = pickle.load(f)
    return model, feature_scaler, target_scaler

FEATURES = [
    "Total_Renewable",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "cloudcover",
    "temperature_2m",
    "relativehumidity_2m",
    "windspeed_10m",
    "windspeed_100m",
    "winddirection_10m",
    "sin_time",
    "cos_time",
    "sin_doy",
    "cos_doy"
]

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <small>GREENGRID • RENEWABLE ENERGY INTELLIGENCE</small>
    <h1>⚡ Renewable Energy Forecast</h1>
    <p>One-hour-ahead renewable generation prediction with an illustrative EV charging equivalent.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- CHECK FILES ----------------
missing = [
    p.name for p in [MODEL_PATH, FEATURE_SCALER_PATH, TARGET_SCALER_PATH]
    if not p.exists()
]

if DATA_PATH is None:
    st.error("Dataset CSV not found in the GitHub repository. Please upload the project dataset.")
    st.stop()

if missing:
    st.error("Missing model files: " + ", ".join(missing))
    st.info("Upload the final 1-hour model and both FINAL_1H scaler files to the repository root.")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
    model, feature_scaler, target_scaler = load_model_and_scalers()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

missing_features = [c for c in FEATURES if c not in df.columns]
if missing_features:
    st.error("Dataset is missing required model features: " + ", ".join(missing_features))
    st.stop()

# Keep only rows with complete model inputs
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=FEATURES).reset_index(drop=True)

if len(df) < 25:
    st.error("Not enough valid hourly records for a 24-hour forecast window.")
    st.stop()

# ---------------- FORECAST CONTROLS ----------------
st.subheader("Choose forecast time")
st.caption("Choose an hour from the dataset. The model uses the previous 24 hours to predict the following hour.")

min_date = df["datetime"].min().date()
max_date = df["datetime"].max().date()

c1, c2 = st.columns([1, 1])
with c1:
    chosen_date = st.date_input(
        "Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

day_rows = df[df["datetime"].dt.date == chosen_date].copy()

if day_rows.empty:
    st.warning("No data is available for this date.")
    st.stop()

with c2:
    available_times = day_rows["datetime"].dt.strftime("%H:%M").tolist()
    chosen_time = st.selectbox("Hour", available_times, index=len(available_times) - 1)

selected_dt = pd.to_datetime(f"{chosen_date} {chosen_time}")
matching = df.index[df["datetime"] == selected_dt].tolist()

if not matching:
    st.error("Selected date and time could not be found in the dataset.")
    st.stop()

row_index = matching[0]

if row_index < 23:
    st.warning("There are not 24 previous hourly records available for this selection. Choose a later hour.")
    st.stop()

st.markdown("")

if st.button("⚡ Generate 1-Hour Forecast"):
    try:
        window = df.loc[row_index - 23:row_index, FEATURES].values
        scaled_window = feature_scaler.transform(window)
        model_input = scaled_window.reshape(1, 24, len(FEATURES))

        scaled_prediction = model.predict(model_input, verbose=0)
        prediction = float(
            target_scaler.inverse_transform(
                np.array(scaled_prediction).reshape(-1, 1)
            )[0, 0]
        )

        prediction = max(0.0, prediction)
        forecast_time = selected_dt + timedelta(hours=1)

        # Illustrative only: assumes forecast value is kWh and EV battery need is 25 kWh.
        ev_equivalent = prediction / 25.0

        st.success(f"Forecast generated for {forecast_time.strftime('%d %b %Y, %H:%M')}")

        st.markdown("### Your forecast")
        a, b = st.columns(2)

        with a:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted renewable generation • next 1 hour</div>
                <div class="metric-value">{prediction:,.2f}</div>
                <div class="metric-note">Value shown in the dataset's original unit</div>
            </div>
            """, unsafe_allow_html=True)

        with b:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Illustrative EV charging equivalent</div>
                <div class="metric-value">{ev_equivalent:,.1f} EVs</div>
                <div class="metric-note">Assumes 25 kWh per EV; illustrative, not a real charging guarantee</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Recent renewable generation")
        history = df.loc[row_index - 23:row_index, ["datetime", "Total_Renewable"]].set_index("datetime")
        st.line_chart(history)

        if "Solar_Generation" in df.columns and "Wind_Generation" in df.columns:
            st.markdown("### Recent solar and wind mix")
            mix = df.loc[row_index - 23:row_index, ["datetime", "Solar_Generation", "Wind_Generation"]]
            mix = mix.set_index("datetime")
            st.area_chart(mix)

        st.caption(
            "The model predicts total renewable generation only. "
            "Solar and wind charts above show historical data, not separate future predictions. "
            "EV equivalence depends on the dataset's unit and the assumed 25 kWh per vehicle."
        )

    except Exception as e:
        st.error(f"Forecast failed: {e}")

st.divider()
st.caption("GreenGrid | GRU-based 1-hour renewable generation forecasting prototype")
