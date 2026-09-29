import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import timedelta
import tensorflow as tf

st.set_page_config(page_title="Green Grid | Renewable Energy Forecast", page_icon="⚡", layout="wide")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp {
  background:
    radial-gradient(ellipse at 12% 0%, rgba(91,105,180,.24), transparent 36%),
    radial-gradient(ellipse at 92% 12%, rgba(49,150,172,.15), transparent 32%),
    #090e1b;
  color:#f4f6ff;
}
[data-testid="stHeader"] { background:rgba(9,14,27,.86); }
.block-container { max-width:1120px; padding:2rem 1.25rem 3.5rem; }
h1,h2,h3,p,label,span { color:#f4f6ff; }
.hero {
  position:relative; overflow:hidden; padding:38px 38px 35px; margin-bottom:25px;
  border:1px solid #35415e; border-radius:25px;
  background:linear-gradient(125deg,rgba(31,39,67,.98),rgba(16,30,48,.98));
  box-shadow:0 18px 55px rgba(0,0,0,.22);
}
.hero:after {
  content:""; position:absolute; width:230px;height:230px;right:-75px;top:-115px;
  border-radius:50%;border:1px solid rgba(148,174,255,.2);
  box-shadow:0 0 0 26px rgba(148,174,255,.035),0 0 0 52px rgba(148,174,255,.025);
  pointer-events:none;
}
.brand { font-size:clamp(54px,8vw,78px); line-height:1; font-weight:900; letter-spacing:-3.5px; color:#fff; }
.brand .bolt { color:#91b4ff; }
.subtitle { font-size:clamp(21px,3.4vw,29px);font-weight:700;color:#d8e1f5;margin-top:15px;letter-spacing:-.45px; }
.intro { color:#aab7d1;font-size:15px;line-height:1.7;margin-top:11px;max-width:760px; }
.eyebrow { color:#a9bbdf;font-size:12px;font-weight:800;letter-spacing:1.8px;text-transform:uppercase; }
.setup {
  border:1px solid #34415e;border-radius:21px;padding:25px 27px 21px;
  background:linear-gradient(145deg,#151e32,#111a2b);margin-bottom:24px;
}
.setup-copy {color:#aebbd4;font-size:14px;margin:7px 0 20px;}
div[data-testid="stDateInput"] label,div[data-testid="stSelectbox"] label {
  color:#f0f3ff!important;font-size:17px!important;font-weight:750!important;margin-bottom:9px!important;
}
div[data-testid="stDateInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"]>div {
  background:#202d47!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;
  border:1.5px solid #7188b5!important;border-radius:14px!important;min-height:68px!important;
  font-size:22px!important;font-weight:700!important;opacity:1!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04);
}
div[data-testid="stDateInput"] input {padding:12px 16px!important;}
div[data-testid="stDateInput"] input::-webkit-calendar-picker-indicator {filter:invert(1);opacity:1;cursor:pointer;}
div[data-testid="stDateInput"] input {color-scheme:dark!important;}
div[data-testid="stDateInput"] input::placeholder {color:#e1e8f7!important;opacity:1!important;}
div[data-testid="stDateInput"] svg,div[data-testid="stSelectbox"] svg {color:#e6edff!important;}
.stButton>button {
  background:linear-gradient(100deg,#9ab0ff,#7bd9e4);color:#0c1426;border:0;
  border-radius:13px;font-size:18px;font-weight:850;min-height:60px;
  box-shadow:0 8px 25px rgba(104,143,220,.2);transition:transform .18s,filter .18s;
}
.stButton>button:hover {filter:brightness(1.08);color:#0c1426;border:0;transform:translateY(-1px);}
.result-card {
  border:1px solid #465b82;border-radius:21px;padding:26px 28px 24px;
  background:linear-gradient(140deg,#202e4b,#17243b);box-shadow:0 14px 34px rgba(0,0,0,.18);
  min-height:208px;animation:rise .55s ease-out both;
}
.result-card.ev {background:linear-gradient(140deg,#20384c,#172c3d);border-color:#3b6977;animation-delay:.1s;}
@keyframes rise {from{opacity:0;transform:translateY(13px)}to{opacity:1;transform:translateY(0)}}
.result-icon {font-size:28px;margin-bottom:14px;}
.result-label {color:#c0cde8;font-size:12px;font-weight:800;letter-spacing:1.4px;}
.result-number {color:#b9caff;font-size:clamp(38px,5.4vw,58px);font-weight:900;letter-spacing:-1.8px;line-height:1.15;margin:15px 0 10px;overflow-wrap:anywhere;}
.ev .result-number {color:#91e5df;}
.result-note {color:#b5c2d8;font-size:13px;line-height:1.6;}
.panel {background:#141e31;border:1px solid #2e3d5a;border-radius:18px;padding:20px 22px;margin-top:20px;}
.smallnote {color:#9baac4;font-size:12px;}
div[data-testid="stDataFrame"] {border:1px solid #34435f;border-radius:12px;}
hr {border-color:#2d3b57;}
@media(max-width:650px) {
 .block-container{padding:1rem .8rem 2rem;}
 .hero{padding:27px 21px;border-radius:20px;}
 .brand{letter-spacing:-2.5px;}
 .setup{padding:20px 17px;}
 .result-card{padding:22px 19px;min-height:0;}
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
  <div class="brand">Green Grid <span class="bolt">⚡</span></div>
  <div class="subtitle">Renewable Energy Forecasting</div>
  <div class="intro">Select a date and time, then generate a one-hour-ahead renewable energy forecast with an illustrative EV charging equivalent.</div>
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

st.markdown('<div class="setup"><div class="eyebrow">01 / Choose forecast time</div><div class="setup-copy">Choose the reference date first, then select the hour to forecast the next hour.</div>', unsafe_allow_html=True)
min_date = df.datetime.min().date()
max_date = df.datetime.max().date()
chosen_date = st.date_input("Choose date", value=max_date, min_value=min_date, max_value=max_date, format="DD MMM YYYY")
day = df[df.datetime.dt.date == chosen_date]
if day.empty:
    st.warning("No dataset records are available on this date.")
    st.stop()
time_options = day.datetime.dt.strftime("%H:%M").tolist()
chosen_time = st.selectbox("Choose time", time_options, index=len(time_options)-1)
generate = st.button("⚡  Generate forecast", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown('<div class="eyebrow">02 / Forecast result</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-card">
      <div class="result-icon">☀️ &nbsp; 🌬️</div>
      <div class="result-label">PREDICTED RENEWABLE GENERATION · NEXT HOUR</div>
      <div class="result-number">{result["value"]:,.2f}</div>
      <div class="result-note">Forecast time: {result["time"]}<br>Value shown in the dataset's original unit.</div>
    </div>
    <div style="height:15px"></div>
    <div class="result-card ev">
      <div class="result-icon">🚙 &nbsp; 🔋</div>
      <div class="result-label">ILLUSTRATIVE EV CHARGING EQUIVALENT</div>
      <div class="result-number">{result["value"]/25:,.1f} EVs</div>
      <div class="result-note">Assumes 25 kWh per EV; illustrative comparison only, not a measured vehicle count or charging guarantee.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="eyebrow">Generation trend</div><p style="color:#aebbd4;font-size:13px">Historical renewable generation across the 24 input hours, with the model estimate shown at the next hour.</p></div>', unsafe_allow_html=True)
    plot = df.loc[max(0, idx-23):idx, ["datetime", "Total_Renewable"]].copy()
    plot = plot.rename(columns={"datetime":"Time", "Total_Renewable":"Observed"})
    future = pd.DataFrame({"Time":[selected_dt+timedelta(hours=1)], "Observed":[np.nan], "Forecast":[result["value"]]})
    plot["Forecast"] = np.nan
    chart = pd.concat([plot, future], ignore_index=True).set_index("Time")
    st.line_chart(chart, use_container_width=True)
    st.caption("Observed values come from the historical dataset. The final point is a one-hour-ahead model estimate.")

    with st.expander("☀️ Solar and 🌬️ wind resource indicators"):
        st.write("These are weather/resource indicators, not separate solar and wind generation predictions.")
        window = df.loc[idx-23:idx]
        left, right = st.columns(2)
        with left:
            st.markdown("**☀️ Shortwave radiation**")
            st.line_chart(window.set_index("datetime")[["shortwave_radiation"]], use_container_width=True)
        with right:
            st.markdown("**🌬️ Wind speed at 10 m**")
            st.line_chart(window.set_index("datetime")[["windspeed_10m"]], use_container_width=True)
else:
    st.info("Choose a date and time, then press **Generate forecast** to see the result.")

st.divider()
st.markdown('<div class="eyebrow">03 / Recent forecasts</div>', unsafe_allow_html=True)
st.caption("Forecasts created during this browser session, newest first.")
if st.session_state.forecast_history:
    h = pd.DataFrame(st.session_state.forecast_history)
    h = h.rename(columns={"reference":"Reference hour", "time":"Predicted hour", "value":"Predicted generation"})
    h["Predicted generation"] = h["Predicted generation"].map(lambda v: f"{v:,.2f}")
    st.dataframe(h, use_container_width=True, hide_index=True)
else:
    st.caption("Your forecast history will appear here after your first prediction.")

with st.expander("How Green Grid works"):
    st.markdown("""
- **Input:** the 24 hourly records ending at your selected reference time.
- **Model:** saved GRU model with its feature and target scalers.
- **Forecast horizon:** one hour ahead.
- **EV equivalent:** illustrative calculation using an assumed 25 kWh per EV.
- **Data source:** historical dataset, not a live grid feed.
""")
st.markdown('<div class="smallnote">Green Grid · GRU-based renewable generation forecasting prototype</div>', unsafe_allow_html=True)
