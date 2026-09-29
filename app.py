import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import timedelta
import tensorflow as tf

st.set_page_config(page_title="GreenGrid | Renewable Forecast", page_icon="⚡", layout="centered")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp {
  background:
    radial-gradient(ellipse at 50% -12%, rgba(28,132,82,.13), transparent 42%),
    #090b0d;
  color:#edf4ef;
}
[data-testid="stHeader"] { background:rgba(9,11,13,.92); }
.block-container { max-width:760px; padding:1.35rem 1.1rem 3rem; }
h1,h2,h3,p,label,span { color:#edf4ef; }
.hero {
  padding:26px 27px 25px; margin:0 0 17px;
  border:1px solid #252e2b; border-radius:20px;
  background:linear-gradient(145deg,#151a18,#101413);
  box-shadow:0 16px 42px rgba(0,0,0,.22);
  animation:fadeUp .55s ease-out both;
}
.hero-top {display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:25px;}
.pill {display:inline-block;background:#123323;color:#70dda0;border:1px solid #20583b;border-radius:30px;padding:6px 12px;font-size:11px;font-weight:800;letter-spacing:.7px;}
.live {color:#a4b0aa;font-size:11px;letter-spacing:1.1px;font-weight:700;}
.brand {font-size:clamp(38px,7vw,52px);line-height:1.08;font-weight:800;letter-spacing:-2.1px;color:#f4f7f5;}
.brand .bolt {color:#72dfa2;}
.subtitle {font-size:clamp(19px,3.5vw,25px);font-weight:550;color:#b8c2bc;margin-top:17px;letter-spacing:-.5px;}
.intro {color:#8f9a93;font-size:14px;line-height:1.75;margin-top:13px;max-width:610px;}
.section-title {color:#b8c5bd;font-size:11px;font-weight:800;letter-spacing:1.6px;text-transform:uppercase;}
.section-copy {color:#849188;font-size:12px;line-height:1.55;margin:5px 0 17px;}
div[data-testid="stVerticalBlockBorderWrapper"] {
  background:linear-gradient(145deg,#151a18,#111514)!important;
  border:1px solid #29332e!important;border-radius:19px!important;
  padding:19px 20px 16px!important;
}
div[data-testid="stDateInput"] label,div[data-testid="stSelectbox"] label {
  color:#bfc9c2!important;font-size:13px!important;font-weight:600!important;margin-bottom:6px!important;
}
div[data-testid="stDateInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"]>div {
  background:#222927!important;color:#f0f5f1!important;-webkit-text-fill-color:#f0f5f1!important;
  border:1px solid #3b4741!important;border-radius:12px!important;min-height:51px!important;
  font-size:17px!important;font-weight:650!important;opacity:1!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.025);
}
div[data-testid="stDateInput"] input {padding:8px 13px!important;color-scheme:dark!important;}
div[data-testid="stDateInput"] input::-webkit-calendar-picker-indicator {filter:invert(.8);opacity:1;cursor:pointer;}
div[data-testid="stDateInput"] svg,div[data-testid="stSelectbox"] svg {color:#75dda3!important;}
.stButton>button {
  background:linear-gradient(100deg,#70d99a,#65dba9);color:#092116;border:1px solid #7ce5a8;
  border-radius:12px;font-size:16px;font-weight:800;min-height:53px;
  box-shadow:0 7px 24px rgba(70,207,129,.13);transition:transform .2s,filter .2s,box-shadow .2s;
}
.stButton>button:hover {filter:brightness(1.06);color:#092116;border-color:#8aefb2;transform:translateY(-1px);box-shadow:0 10px 28px rgba(70,207,129,.2);}
.stButton>button:active {transform:scale(.99);}
.result-card {
  border:1px solid #29342e;border-radius:19px;padding:23px 23px 21px;
  background:linear-gradient(145deg,#171d1a,#121715);box-shadow:0 14px 34px rgba(0,0,0,.2);
  animation:fadeUp .55s ease-out both;
}
.result-card.ev {background:linear-gradient(145deg,#17231d,#131a16);border-color:#31513d;animation-delay:.09s;}
@keyframes fadeUp {from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.result-top {display:flex;justify-content:space-between;align-items:center;gap:10px;}
.result-label {color:#98a59c;font-size:11px;font-weight:800;letter-spacing:1.2px;}
.result-icon {font-size:21px;color:#76dfa2;}
.result-number {color:#f1f7f2;font-size:clamp(37px,7vw,51px);font-weight:750;letter-spacing:-1.7px;line-height:1.15;margin:17px 0 3px;overflow-wrap:anywhere;}
.ev .result-number {color:#82e5aa;}
.unit {color:#9aa79e;font-size:13px;}
.result-note {color:#849188;font-size:12px;line-height:1.6;margin-top:12px;}
.ev-row {display:flex;justify-content:space-between;align-items:center;gap:12px;border-top:1px solid #303b34;margin-top:18px;padding-top:16px;}
.ev-count {color:#edf6ef;font-size:23px;font-weight:750;}
.ev-assumption {color:#8d9a91;font-size:12px;text-align:right;}
.history {border:1px solid #29342e;border-radius:16px;background:#121715;padding:17px 18px;margin-top:18px;}
.smallnote {color:#758179;font-size:11px;}
div[data-testid="stDataFrame"] {border:1px solid #29342e;border-radius:12px;}
hr {border-color:#29342e;}
@media(max-width:600px) {
 .block-container{padding:1rem .8rem 2rem;}
 .hero{padding:22px 20px;border-radius:18px;}
 .hero-top{margin-bottom:22px;}
 .brand{letter-spacing:-1.7px;}
 div[data-testid="stVerticalBlockBorderWrapper"]{padding:16px 14px 13px!important;}
 .result-card{padding:20px 18px;}
}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;}}
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

FEATURES = ["Total_Renewable", "shortwave_radiation", "direct_radiation", "diffuse_radiation",
            "cloudcover", "temperature_2m", "relativehumidity_2m", "windspeed_10m",
            "windspeed_100m", "winddirection_10m", "sin_time", "cos_time", "sin_doy", "cos_doy"]

@st.cache_data
def load_data(path):
    d = pd.read_csv(path)
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    d = d.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    if "sin_time" not in d:
        h = d.datetime.dt.hour
        d["sin_time"] = np.sin(2*np.pi*h/24)
        d["cos_time"] = np.cos(2*np.pi*h/24)
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
  <div class="hero-top"><span class="pill">GREEN ENERGY INTELLIGENCE</span><span class="live">● &nbsp; FORECAST CONSOLE</span></div>
  <div class="brand">GreenGrid <span class="bolt">⚡</span></div>
  <div class="subtitle">Renewable Energy Forecasting</div>
  <div class="intro">Predict the next hour of renewable generation using historical energy patterns.</div>
</div>
""", unsafe_allow_html=True)

missing = [p.name for p in [MODEL_PATH, FEATURE_SCALER_PATH, TARGET_SCALER_PATH] if not p.exists()]
if DATA_PATH is None:
    st.error("Dataset CSV not found in the repository.")
    st.stop()
if missing:
    st.error("Missing model assets: " + ", ".join(missing))
    st.stop()
try:
    df = load_data(str(DATA_PATH))
    model, feature_scaler, target_scaler = load_assets()
except Exception as e:
    st.error(f"Project files could not be loaded: {e}")
    st.stop()

missing_cols = [c for c in FEATURES if c not in df]
if missing_cols:
    st.error("Dataset is missing required features: " + ", ".join(missing_cols))
    st.stop()
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=FEATURES).reset_index(drop=True)
if len(df) < 25:
    st.error("Not enough valid hourly records.")
    st.stop()

st.markdown('<div class="section-title">01 / Select forecast time</div><div class="section-copy">Choose a date and reference hour. GreenGrid estimates renewable generation for the following hour.</div>', unsafe_allow_html=True)
with st.container(border=True):
    min_date = df.datetime.min().date()
    max_date = df.datetime.max().date()
    chosen_date = st.date_input("Select date", value=max_date, min_value=min_date, max_value=max_date, format="DD/MM/YYYY")
    day = df[df.datetime.dt.date == chosen_date]
    if day.empty:
        st.warning("No dataset records are available on this date.")
        st.stop()
    time_options = day.datetime.dt.strftime("%H:%M").tolist()
    chosen_time = st.selectbox("Select time", time_options, index=len(time_options)-1)
    generate = st.button("⚡  Generate Forecast  ↗", use_container_width=True)

selected_dt = pd.to_datetime(f"{chosen_date} {chosen_time}")
found = df.index[df.datetime == selected_dt].tolist()
if not found:
    st.error("Selected timestamp is not available.")
    st.stop()
idx = found[0]
if idx < 23:
    st.warning("Choose a later hour; the model needs 24 hourly records.")
    st.stop()

if "forecast_history" not in st.session_state:
    st.session_state.forecast_history = []
if generate:
    try:
        with st.spinner("Reading recent energy patterns and generating forecast…"):
            x = df.loc[idx-23:idx, FEATURES].values
            x = feature_scaler.transform(x).reshape(1, 24, len(FEATURES))
            y = model.predict(x, verbose=0)
            pred = max(0.0, float(target_scaler.inverse_transform(np.asarray(y).reshape(-1,1))[0,0]))
        next_time = selected_dt + timedelta(hours=1)
        item = {"reference": selected_dt.strftime("%d %b %Y · %H:%M"),
                "time": next_time.strftime("%d %b %Y · %H:%M"), "value": pred}
        st.session_state["latest_forecast"] = item
        st.session_state.forecast_history.insert(0, item)
        st.session_state.forecast_history = st.session_state.forecast_history[:10]
    except Exception as e:
        st.error(f"Prediction failed: {e}")

result = st.session_state.get("latest_forecast")
if result:
    st.markdown('<div class="section-title" style="margin:21px 0 10px">02 / Forecast result</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-card">
      <div class="result-top"><div class="result-label">TOTAL RENEWABLE GENERATION</div><div class="result-icon">ϟ</div></div>
      <div class="result-number">{result["value"]:,.2f}</div>
      <div class="unit">kWh · dataset-scale estimate</div>
      <div class="result-note">Forecast timestamp: {result["time"]}<br>One-hour-ahead model estimate, not a live grid measurement.</div>
    </div>
    <div style="height:12px"></div>
    <div class="result-card ev">
      <div class="result-top"><div class="result-label">ILLUSTRATIVE EV CHARGING EQUIVALENT</div><div class="result-icon">⌁</div></div>
      <div class="ev-row" style="border-top:0;margin-top:15px;padding-top:0">
        <div class="ev-count">{result["value"]/25:,.1f} EVs</div>
        <div class="ev-assumption">25 kWh / EV</div>
      </div>
      <div class="result-note">Illustrative comparison only; not a measured vehicle count or charging guarantee.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title" style="margin:24px 0 8px">03 / Forecast history</div>', unsafe_allow_html=True)
st.caption("Predictions generated during this browser session, newest first.")
if st.session_state.forecast_history:
    h = pd.DataFrame(st.session_state.forecast_history)
    h = h.rename(columns={"reference":"Reference hour", "time":"Predicted hour", "value":"Predicted generation"})
    h["Predicted generation"] = h["Predicted generation"].map(lambda v: f"{v:,.2f}")
    st.dataframe(h, use_container_width=True, hide_index=True)
else:
    st.markdown('<div class="history"><span class="smallnote">Your generated forecast timestamps and values will appear here.</span></div>', unsafe_allow_html=True)

with st.expander("How GreenGrid works"):
    st.markdown("""
- **Input:** 24 hourly dataset records ending at the selected reference time.
- **Model:** saved GRU model with its feature and target scalers.
- **Forecast horizon:** one hour ahead.
- **EV equivalent:** illustrative calculation using an assumed 25 kWh per EV.
- **Data source:** historical dataset, not a live grid feed.
""")
st.markdown('<div class="smallnote" style="text-align:center;padding:18px 0 4px">GREENGRID · GRU-BASED RENEWABLE GENERATION FORECASTING</div>', unsafe_allow_html=True)
