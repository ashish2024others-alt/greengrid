import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import timedelta
import tensorflow as tf

st.set_page_config(page_title="GreenGrid | Renewable Energy Forecasting", page_icon="⚡", layout="centered")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp {
  background: radial-gradient(ellipse at 50% -15%, rgba(27,103,143,.20), transparent 48%), #080d16;
  color:#edf4fb;
}
[data-testid="stHeader"] { background:rgba(8,13,22,.94); }
.block-container { max-width:820px; padding:1.4rem 1.1rem 3.2rem; }
h1,h2,h3,p,label,span { color:#edf4fb; }
.hero {
  padding:30px 28px 27px; margin:0 0 23px;
  border:1px solid #263b50; border-radius:23px;
  background:linear-gradient(145deg,#142438,#101a2a 70%,#111c2a);
  box-shadow:0 18px 48px rgba(0,0,0,.28);
  animation:rise .65s ease-out both;
}
.eyebrow { color:#70c7e8; font-size:11px; font-weight:800; letter-spacing:2px; text-transform:uppercase; margin-bottom:17px; }
.brand { color:#f4f8ff; font-size:clamp(48px,10vw,68px); line-height:1.02; font-weight:900; letter-spacing:-2.6px; }
.brand-mark { color:#70c7e8; }
.subtitle { color:#b8cce0; font-size:clamp(19px,4vw,25px); font-weight:700; margin-top:13px; }
.intro { color:#91a7bb; font-size:14px; line-height:1.7; margin-top:10px; max-width:640px; }
.section-head { color:#9fc3dd; font-size:11px; font-weight:850; letter-spacing:1.7px; text-transform:uppercase; margin:22px 0 6px; }
.section-help { color:#8196aa; font-size:13px; line-height:1.6; margin-bottom:13px; }
div[data-testid="stVerticalBlockBorderWrapper"] {
  background:linear-gradient(145deg,#121f30,#101a28)!important;
  border:1px solid #2a4055!important; border-radius:20px!important;
  padding:21px 21px 17px!important;
}
div[data-testid="stDateInput"] label,div[data-testid="stSelectbox"] label {
  color:#c2d4e5!important; font-size:14px!important; font-weight:700!important; margin-bottom:8px!important;
}
div[data-testid="stDateInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"]>div {
  background:#1a2a3c!important; color:#f5f9ff!important; -webkit-text-fill-color:#f5f9ff!important;
  border:1px solid #3b5872!important; border-radius:13px!important; min-height:58px!important;
  font-size:19px!important; font-weight:700!important; opacity:1!important;
}
div[data-testid="stDateInput"] input { padding:10px 14px!important; color-scheme:dark!important; }
div[data-testid="stDateInput"] input::-webkit-calendar-picker-indicator { filter:invert(.85); opacity:1; cursor:pointer; }
div[data-testid="stDateInput"] svg,div[data-testid="stSelectbox"] svg { color:#70c7e8!important; }
.stButton>button {
  background:linear-gradient(100deg,#36a6d2,#55c5d9); color:#061724; border:1px solid #75d5e8;
  border-radius:14px; font-size:17px; font-weight:850; min-height:56px;
  box-shadow:0 8px 25px rgba(42,164,207,.20); transition:transform .2s,filter .2s,box-shadow .2s;
}
.stButton>button:hover { filter:brightness(1.08); color:#061724; border-color:#9ce8f2; transform:translateY(-2px); box-shadow:0 12px 30px rgba(42,164,207,.27); }
.result-card {
  border:1px solid #2c4a62; border-radius:21px; padding:23px 23px 20px;
  background:linear-gradient(145deg,#14263a,#101b2a); box-shadow:0 15px 38px rgba(0,0,0,.24);
  animation:rise .65s ease-out both;
}
.result-card.ev { background:linear-gradient(145deg,#142c38,#10212d); border-color:#2e6574; animation-delay:.1s; }
.result-top { display:flex; justify-content:space-between; align-items:center; gap:12px; }
.result-label { color:#9db4c9; font-size:11px; font-weight:850; letter-spacing:1.35px; }
.result-icon { font-size:27px; animation:float 2.8s ease-in-out infinite; }
.result-number { color:#f2f8ff; font-size:clamp(38px,8vw,53px); font-weight:850; letter-spacing:-1.8px; line-height:1.15; margin:17px 0 4px; overflow-wrap:anywhere; }
.ev .result-number { color:#8ce0e8; }
.unit { color:#a0b5c8; font-size:13px; }
.result-note { color:#8199ad; font-size:12px; line-height:1.65; margin-top:13px; }
.history-box { border:1px solid #2a4055; border-radius:17px; background:#101b29; padding:17px 18px; margin-top:12px; }
.small-note { color:#8095a8; font-size:11px; line-height:1.6; }
div[data-testid="stDataFrame"] { border:1px solid #2a4055; border-radius:12px; overflow:hidden; }
div[data-testid="stAlert"] { border-radius:12px; }
@keyframes rise { from {opacity:0;transform:translateY(13px)} to {opacity:1;transform:translateY(0)} }
@keyframes float { 0%,100% {transform:translateY(0)} 50% {transform:translateY(-4px)} }
@media(max-width:600px) {
 .block-container { padding:1rem .8rem 2.2rem; }
 .hero { padding:24px 20px; border-radius:19px; }
 div[data-testid="stVerticalBlockBorderWrapper"] { padding:17px 14px 14px!important; }
 .result-card { padding:20px 18px; }
}
@media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none!important; transition:none!important; } }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent
MODEL_PATH = BASE / "renewable_gru_model_FINAL_1H.keras"
FEATURE_SCALER_PATH = BASE / "feature_scaler_FINAL_1H.pkl"
TARGET_SCALER_PATH = BASE / "target_scaler_FINAL_1H.pkl"
possible_csvs = [BASE / "renewable_energy_project_dataset.csv", BASE / "reg_forecasting_v2_project_clean.csv"]
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

FEATURES = [
    "Total_Renewable", "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "cloudcover", "temperature_2m", "relativehumidity_2m", "windspeed_10m",
    "windspeed_100m", "winddirection_10m", "sin_time", "cos_time", "sin_doy", "cos_doy"
]

@st.cache_data
def load_data(path):
    d = pd.read_csv(path)
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    d = d.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    if "sin_time" not in d:
        hour = d.datetime.dt.hour
        d["sin_time"] = np.sin(2*np.pi*hour/24)
        d["cos_time"] = np.cos(2*np.pi*hour/24)
    if "sin_doy" not in d:
        day = d.datetime.dt.dayofyear
        d["sin_doy"] = np.sin(2*np.pi*day/365.25)
        d["cos_doy"] = np.cos(2*np.pi*day/365.25)
    return d

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model, joblib.load(FEATURE_SCALER_PATH), joblib.load(TARGET_SCALER_PATH)

st.markdown("""
<div class="hero">
  <div class="eyebrow">RENEWABLE ENERGY • FORECASTING SYSTEM</div>
  <div class="brand">Renewable Energy <span class="brand-mark">Forecasting</span></div>
  <div class="subtitle">One-hour-ahead generation & EV charging estimation</div>
  <div class="intro">Choose a date and reference time to estimate the next hour’s renewable generation, with an illustrative EV energy equivalent.</div>
</div>
""", unsafe_allow_html=True)

missing_assets = [p.name for p in [MODEL_PATH, FEATURE_SCALER_PATH, TARGET_SCALER_PATH] if not p.exists()]
if DATA_PATH is None:
    st.error("Dataset CSV not found in the repository.")
    st.stop()
if missing_assets:
    st.error("Missing model files: " + ", ".join(missing_assets))
    st.stop()

try:
    df = load_data(str(DATA_PATH))
    model, feature_scaler, target_scaler = load_assets()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

missing_cols = [c for c in FEATURES if c not in df.columns]
if missing_cols:
    st.error("Dataset is missing required features: " + ", ".join(missing_cols))
    st.stop()

df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=FEATURES).reset_index(drop=True)
if len(df) < 25:
    st.error("Not enough valid hourly records to create a forecast.")
    st.stop()

st.markdown('<div class="section-head">Choose your forecast</div><div class="section-help">Select the date first, then choose the reference time. GreenGrid will predict the following hour.</div>', unsafe_allow_html=True)
with st.container(border=True):
    min_date = df.datetime.min().date()
    max_date = df.datetime.max().date()
    chosen_date = st.date_input("1. Choose date", value=max_date, min_value=min_date, max_value=max_date, format="DD/MM/YYYY")
    day = df[df.datetime.dt.date == chosen_date]
    if day.empty:
        st.warning("There are no records for this date. Please choose another date.")
        st.stop()
    time_options = day.datetime.dt.strftime("%H:%M").tolist()
    chosen_time = st.selectbox("Choose reference time", time_options, index=len(time_options)-1)
    generate = st.button("⚡  GENERATE FORECAST", use_container_width=True)

selected_dt = pd.to_datetime(f"{chosen_date} {chosen_time}")
found = df.index[df.datetime == selected_dt].tolist()
if not found:
    st.error("The selected timestamp is not available in the dataset.")
    st.stop()
idx = found[0]
if idx < 23:
    st.warning("Choose a later time. The model needs 24 hours of historical records.")
    st.stop()

if "forecast_history" not in st.session_state:
    st.session_state.forecast_history = []

if generate:
    try:
        with st.spinner("Analysing the previous 24 hours…"):
            x = df.loc[idx-23:idx, FEATURES].values
            x_scaled = feature_scaler.transform(x).reshape(1, 24, len(FEATURES))
            y_scaled = model.predict(x_scaled, verbose=0)
            prediction = float(target_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1))[0, 0])
            prediction = max(0.0, prediction)
        next_dt = selected_dt + timedelta(hours=1)
        result = {
            "reference": selected_dt.strftime("%d %b %Y · %H:%M"),
            "time": next_dt.strftime("%d %b %Y · %H:%M"),
            "value": prediction
        }
        st.session_state.latest_forecast = result
        st.session_state.forecast_history.insert(0, result)
        st.session_state.forecast_history = st.session_state.forecast_history[:10]
    except Exception as e:
        st.error(f"Forecast could not be generated: {e}")

result = st.session_state.get("latest_forecast")
if result:
    st.markdown('<div class="section-head">02 / Your forecast</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-card">
      <div class="result-top"><div class="result-label">☀️  TOTAL RENEWABLE GENERATION</div><div class="result-icon">🌞</div></div>
      <div class="result-number">{result["value"]:,.2f}</div>
      <div class="unit">Value in the dataset’s original unit</div>
      <div class="result-note">Predicted hour: {result["time"]}<br>One-hour-ahead model estimate based on historical data — not a live grid reading.</div>
    </div>
    <div style="height:13px"></div>
    <div class="result-card ev">
      <div class="result-top"><div class="result-label">🚙  EV CHARGING EQUIVALENT</div><div class="result-icon">🔋</div></div>
      <div class="result-number">{result["value"]/25:,.1f} EVs</div>
      <div class="unit">Estimated equivalent using the project assumption of 25 kWh per EV</div>
      <div class="result-note">Illustrative comparison only. Dataset units are not specified as a physical energy unit, so this is not a measured vehicle count or a charging guarantee.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">03 / Recent generation pattern</div>', unsafe_allow_html=True)
    recent = df.loc[max(0, idx-23):idx, ["datetime", "Total_Renewable"]].copy()
    recent = recent.set_index("datetime").rename(columns={"Total_Renewable": "Historical generation"})
    forecast_point = pd.DataFrame({"Historical generation": [np.nan]}, index=[selected_dt + timedelta(hours=1)])
    forecast_point["Next-hour forecast"] = result["value"]
    recent["Next-hour forecast"] = np.nan
    graph = pd.concat([recent, forecast_point])
    st.line_chart(graph, height=260)
    st.caption("The line shows the 24 historical hourly values used as model input. The forecast marker represents the predicted next hour.")

st.markdown('<div class="section-head">04 / Forecast history</div>', unsafe_allow_html=True)
st.caption("Forecasts generated during this browser session, newest first.")
if st.session_state.forecast_history:
    history = pd.DataFrame(st.session_state.forecast_history)
    history = history.rename(columns={"reference": "Reference hour", "time": "Predicted hour", "value": "Predicted generation"})
    history["Predicted generation"] = history["Predicted generation"].map(lambda v: f"{v:,.2f}")
    st.dataframe(history, use_container_width=True, hide_index=True)
else:
    st.markdown('<div class="history-box"><span class="small-note">Your forecast date, predicted hour and result will appear here after you generate a forecast.</span></div>', unsafe_allow_html=True)

with st.expander("About this forecasting system"):
    st.markdown("""
- **Input:** 24 consecutive hourly records with 14 selected features.
- **Model:** trained GRU model with saved feature and target scalers.
- **Forecast horizon:** one hour ahead.
- **EV estimate:** an illustrative calculation using the project's 25 kWh per EV assumption.
- **Data:** historical dataset; this dashboard is not connected to live grid measurements.
""")

st.markdown('<div class="small-note" style="text-align:center;padding:23px 0 5px">ONE-HOUR-AHEAD RENEWABLE GENERATION FORECASTING • GRU MODEL</div>', unsafe_allow_html=True)
