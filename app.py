import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import timedelta
import tensorflow as tf

st.set_page_config(page_title="GreenGrid | Renewable Energy Forecasting", page_icon="⚡", layout="wide")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp {
  background:
    radial-gradient(ellipse at 15% 0%, rgba(72, 83, 145, .22), transparent 38%),
    radial-gradient(ellipse at 90% 10%, rgba(20, 139, 160, .13), transparent 34%),
    #0b1020;
  color: #f2f4fb;
}
[data-testid="stHeader"] { background: rgba(11,16,32,.88); }
.block-container { max-width: 1080px; padding-top: 2.2rem; padding-bottom: 3rem; }
h1,h2,h3,p,label,span { color: #f2f4fb; }
.brand-title {
  font-size: clamp(42px, 7vw, 66px); font-weight: 850; letter-spacing: -2.5px;
  line-height: 1.02; margin: 0; color: #f7f8ff;
}
.brand-title .spark { color: #8ea8ff; }
.subtitle {
  margin-top: 13px; font-size: clamp(19px, 3vw, 25px); font-weight: 650;
  color: #c6d0e8; letter-spacing: -.4px;
}
.intro { color: #9eabc5; font-size: 15px; line-height: 1.65; margin-top: 9px; max-width: 720px; }
.hero {
  padding: 31px 34px 32px; border: 1px solid #303d5c; border-radius: 24px;
  background: linear-gradient(125deg, rgba(31,40,68,.96), rgba(17,29,48,.96));
  box-shadow: 0 18px 55px rgba(0,0,0,.23); margin-bottom: 27px;
  position: relative; overflow: hidden;
}
.hero:after {
  content:""; position:absolute; width:220px; height:220px; right:-70px; top:-105px;
  border-radius:50%; border:1px solid rgba(143,166,255,.16);
  box-shadow:0 0 0 24px rgba(143,166,255,.025),0 0 0 48px rgba(143,166,255,.02);
  pointer-events:none;
}
.section-label { color:#aab9dc; font-size:12px; font-weight:750; letter-spacing:2px; text-transform:uppercase; }
.setup-card {
  background: linear-gradient(145deg,#171f34,#121a2c); border:1px solid #2c3957;
  border-radius:20px; padding:24px 26px 20px; margin-bottom:24px;
}
div[data-testid="stDateInput"] label, div[data-testid="stSelectbox"] label {
  color:#dbe3f7 !important; font-size:16px !important; font-weight:650 !important;
  margin-bottom:8px !important;
}
div[data-testid="stDateInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background:#202b43 !important; color:#ffffff !important; border:1px solid #526488 !important;
  border-radius:12px !important; min-height:54px !important; font-size:19px !important;
}
div[data-testid="stDateInput"] input { color-scheme:dark; }
div[data-testid="stDateInput"] svg, div[data-testid="stSelectbox"] svg { color:#dce5ff !important; }
.stButton>button {
  background:linear-gradient(100deg,#91a8ff,#77d5e5); color:#10182b; border:0;
  border-radius:13px; font-size:17px; font-weight:800; min-height:54px;
  box-shadow:0 8px 24px rgba(99,139,220,.18); transition:transform .18s,filter .18s;
}
.stButton>button:hover { filter:brightness(1.08); color:#10182b; border:0; transform:translateY(-1px); }
.result-card {
  border:1px solid #3c5275; border-radius:20px; padding:25px 26px 23px;
  background:linear-gradient(140deg,#202e4b,#17243b);
  box-shadow:0 14px 34px rgba(0,0,0,.18); min-height:205px;
  animation:appear .55s ease-out both;
}
.result-card.ev { background:linear-gradient(140deg,#20364a,#172b3c); border-color:#3b6472; animation-delay:.1s; }
@keyframes appear { from{opacity:0;transform:translateY(13px)} to{opacity:1;transform:translateY(0)} }
.result-label { color:#b8c8e8; font-size:12px; font-weight:750; letter-spacing:1.5px; }
.result-number { color:#b5c7ff; font-size:clamp(35px,5vw,53px); font-weight:850; letter-spacing:-1.6px; line-height:1.15; margin:17px 0 10px; overflow-wrap:anywhere; }
.ev .result-number { color:#8fe2df; }
.result-note { color:#b1bfd5; font-size:13px; line-height:1.55; }
.iconline { font-size:22px; margin-bottom:12px; }
.panel {
  background:#141d30; border:1px solid #2c3955; border-radius:18px;
  padding:20px 22px; margin-top:18px;
}
.smallnote { color:#95a3bc; font-size:12px; }
div[data-testid="stDataFrame"] { border:1px solid #34435f; border-radius:12px; }
hr { border-color:#2c3955; }
@media(max-width:650px) {
  .block-container { padding:1rem .85rem 2rem; }
  .hero { padding:25px 21px; border-radius:19px; }
  .setup-card { padding:19px 17px; }
  .result-card { padding:21px 19px; min-height:0; }
}
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
  <div class="brand-title">GreenGrid <span class="spark">⚡</span></div>
  <div class="subtitle">Renewable Energy Forecasting &amp; EV Outlook</div>
  <div class="intro">Choose a date and reference hour to estimate renewable generation for the following hour, with an illustrative EV charging equivalent.</div>
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

st.markdown('<div class="setup-card"><div class="section-label">01 / Set your forecast</div><p style="color:#aab8d2;margin:6px 0 20px">Pick the last known hour. GreenGrid will estimate the next hour.</p>', unsafe_allow_html=True)
min_date = df.datetime.min().date()
max_date = df.datetime.max().date()
chosen_date = st.date_input("Choose date", value=max_date, min_value=min_date, max_value=max_date, format="DD MMM YYYY")
day = df[df.datetime.dt.date == chosen_date]
if day.empty:
    st.warning("No dataset records are available on this date.")
    st.stop()
time_options = day.datetime.dt.strftime("%H:%M").tolist()
chosen_time = st.selectbox("Choose reference time", time_options, index=len(time_options)-1)
generate = st.button("⚡  Generate forecast", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-label">02 / Your forecast</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-card">
      <div class="iconline">☀️ 🌬️</div>
      <div class="result-label">PREDICTED RENEWABLE GENERATION · NEXT HOUR</div>
      <div class="result-number">{result["value"]:,.2f}</div>
      <div class="result-note">Forecast time: {result["time"]}<br>Value shown in the dataset's original unit.</div>
    </div>
    <div style="height:14px"></div>
    <div class="result-card ev">
      <div class="iconline">🚙 🔋</div>
      <div class="result-label">ILLUSTRATIVE EV CHARGING EQUIVALENT</div>
      <div class="result-number">{result["value"]/25:,.1f} EVs</div>
      <div class="result-note">Assumes 25 kWh per EV. Illustrative comparison only, not a measured vehicle count or charging guarantee.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="section-label">Generation trend</div><p style="color:#aab8d2;font-size:13px">Observed renewable generation in the 24-hour input window, followed by the forecast point.</p></div>', unsafe_allow_html=True)
    plot = df.loc[max(0, idx-23):idx, ["datetime", "Total_Renewable"]].copy()
    plot = plot.rename(columns={"datetime":"Time", "Total_Renewable":"Observed"})
    future = pd.DataFrame({"Time":[selected_dt+timedelta(hours=1)], "Observed":[np.nan], "Forecast":[result["value"]]})
    plot["Forecast"] = np.nan
    chart = pd.concat([plot, future], ignore_index=True).set_index("Time")
    st.line_chart(chart, use_container_width=True)
    st.caption("The forecast is a model estimate for the next hour; the preceding values are historical observations.")

    with st.expander("☀️ Solar and 🌬️ wind resource indicators"):
        st.write("These are weather/resource measurements, not separate solar and wind generation outputs.")
        window = df.loc[idx-23:idx]
        left, right = st.columns(2)
        with left:
            st.markdown("**☀️ Shortwave radiation**")
            st.line_chart(window.set_index("datetime")[["shortwave_radiation"]], use_container_width=True)
        with right:
            st.markdown("**🌬️ Wind speed at 10 m**")
            st.line_chart(window.set_index("datetime")[["windspeed_10m"]], use_container_width=True)
else:
    st.info("Your forecast will appear here after you press **Generate forecast**.")

st.divider()
st.markdown('<div class="section-label">03 / Forecast history</div>', unsafe_allow_html=True)
st.caption("Forecasts created in this browser session, newest first.")
if st.session_state.forecast_history:
    h = pd.DataFrame(st.session_state.forecast_history)
    h = h.rename(columns={"reference":"Reference hour", "time":"Predicted hour", "value":"Predicted generation"})
    h["Predicted generation"] = h["Predicted generation"].map(lambda v: f"{v:,.2f}")
    st.dataframe(h, use_container_width=True, hide_index=True)
else:
    st.caption("No forecasts generated in this session yet.")

with st.expander("How this forecast works"):
    st.markdown("""
- **Input:** 24 hourly records ending at the selected reference time.
- **Model:** saved GRU model with its feature and target scalers.
- **Forecast horizon:** one hour ahead.
- **EV equivalent:** an illustrative calculation using an assumed 25 kWh per EV.
- **Data source:** historical dataset, not a live grid feed.
""")
st.markdown('<div class="smallnote">GreenGrid · GRU-based renewable generation forecasting prototype</div>', unsafe_allow_html=True)
