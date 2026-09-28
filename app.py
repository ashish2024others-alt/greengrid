import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import timedelta
import tensorflow as tf

st.set_page_config(page_title="GreenGrid | Renewable Forecasting",page_icon="⚡",layout="wide")

st.markdown("""
<style>
:root{color-scheme:dark}
.stApp{background:radial-gradient(ellipse at 50% -15%,#28344a 0%,#151c2b 42%,#0d1320 100%);color:#edf1f8}
[data-testid="stHeader"]{background:rgba(13,19,32,.8)}
.block-container{max-width:1200px;padding-top:2rem;padding-bottom:3rem}
h1,h2,h3,p,label,span{color:#edf1f8}
.brand{font-size:12px;letter-spacing:3px;color:#aab8d0;font-weight:700}
.hero{padding:30px 34px 28px;border:1px solid #3b4b68;border-radius:20px;background:linear-gradient(120deg,rgba(40,52,75,.95),rgba(22,32,49,.92));box-shadow:0 16px 48px rgba(0,0,0,.22);margin-bottom:22px}
.hero h1{font-size:clamp(32px,5vw,48px);letter-spacing:-1.5px;margin:10px 0 8px}
.hero p{font-size:16px;color:#bdc9dc;margin:0;max-width:760px}
.eyebrow{font-size:12px;letter-spacing:2px;color:#9fb7df;font-weight:700}
.card{background:linear-gradient(145deg,#202b40,#182337);border:1px solid #3a4b67;border-radius:17px;padding:22px 24px;box-shadow:0 12px 28px rgba(0,0,0,.16);height:100%}
.result{animation:rise .65s ease-out both;border-color:#6488bc;background:linear-gradient(135deg,#263b5a,#1b2a41)}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.kicker{font-size:12px;letter-spacing:1.5px;color:#a9bad5;font-weight:650}
.big{font-size:clamp(32px,4vw,46px);font-weight:800;letter-spacing:-1px;color:#a9c9ff;line-height:1.2;margin:12px 0 6px}
.sub{font-size:13px;color:#b8c5d9}
div[data-testid="stMetric"]{background:#1c293e;border:1px solid #3a4b67;border-radius:13px;padding:15px}
div[data-testid="stMetricLabel"]{color:#b8c5d9}
.stButton>button{background:linear-gradient(90deg,#9fc3ff,#c1d7ff);color:#142238;border:0;border-radius:11px;font-weight:750;min-height:48px}
.stButton>button:hover{background:#d4e3ff;color:#101d31;border:0}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:#1b273b;border-color:#435571}
hr{border-color:#34435b}
[data-testid="stDataFrame"]{border:1px solid #394b66;border-radius:12px}
.smallnote{color:#9eacc2;font-size:12px}
</style>
""",unsafe_allow_html=True)

BASE=Path(__file__).parent
MODEL_PATH=BASE/"renewable_gru_model_FINAL_1H.keras"
FEATURE_SCALER_PATH=BASE/"feature_scaler_FINAL_1H.pkl"
TARGET_SCALER_PATH=BASE/"target_scaler_FINAL_1H.pkl"
possible_csvs=[BASE/"renewable_energy_project_dataset.csv",BASE/"reg_forecasting_v2_project_clean.csv"]
DATA_PATH=next((p for p in possible_csvs if p.exists()),None)
if DATA_PATH is None:
    for p in BASE.glob("*.csv"):
        try:
            cols=pd.read_csv(p,nrows=2).columns
            if "datetime" in cols and "Total_Renewable" in cols: DATA_PATH=p; break
        except Exception: pass

FEATURES=["Total_Renewable","shortwave_radiation","direct_radiation","diffuse_radiation","cloudcover","temperature_2m","relativehumidity_2m","windspeed_10m","windspeed_100m","winddirection_10m","sin_time","cos_time","sin_doy","cos_doy"]

@st.cache_data
def load_data(path):
    d=pd.read_csv(path)
    d["datetime"]=pd.to_datetime(d["datetime"],errors="coerce")
    d=d.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    if "sin_time" not in d:
        h=d.datetime.dt.hour; d["sin_time"]=np.sin(2*np.pi*h/24); d["cos_time"]=np.cos(2*np.pi*h/24)
    if "sin_doy" not in d:
        day=d.datetime.dt.dayofyear; d["sin_doy"]=np.sin(2*np.pi*day/365.25); d["cos_doy"]=np.cos(2*np.pi*day/365.25)
    return d

@st.cache_resource
def load_assets():
    return tf.keras.models.load_model(MODEL_PATH,compile=False),joblib.load(FEATURE_SCALER_PATH),joblib.load(TARGET_SCALER_PATH)

st.markdown('<div class="hero"><div class="brand">⚡ GREENGRID</div><h1>Renewable Energy Forecasting</h1><p>One-hour-ahead generation prediction with an illustrative EV charging equivalent. Choose a date and hour, then generate a forecast.</p></div>',unsafe_allow_html=True)

missing=[p.name for p in [MODEL_PATH,FEATURE_SCALER_PATH,TARGET_SCALER_PATH] if not p.exists()]
if DATA_PATH is None: st.error("Dataset CSV not found in repository."); st.stop()
if missing: st.error("Missing model assets: "+", ".join(missing)); st.stop()
try: df=load_data(str(DATA_PATH)); model,feature_scaler,target_scaler=load_assets()
except Exception as e: st.error(f"Project files could not be loaded: {e}"); st.stop()
missing_cols=[c for c in FEATURES if c not in df]
if missing_cols: st.error("Dataset is missing required features: "+", ".join(missing_cols)); st.stop()
df[FEATURES]=df[FEATURES].apply(pd.to_numeric,errors="coerce")
df=df.dropna(subset=FEATURES).reset_index(drop=True)
if len(df)<25: st.error("Not enough valid hourly records."); st.stop()

st.markdown("### Forecast setup")
a,b,c=st.columns([1,1,1.15],gap="medium")
min_date=df.datetime.min().date(); max_date=df.datetime.max().date()
with a: chosen_date=st.date_input("Choose date",value=max_date,min_value=min_date,max_value=max_date)
day=df[df.datetime.dt.date==chosen_date]
if day.empty: st.warning("No dataset records on this date."); st.stop()
with b: chosen_time=st.selectbox("Choose reference hour",day.datetime.dt.strftime("%H:%M").tolist(),index=len(day)-1)
with c:
    st.markdown("<div style='height:27px'></div>",unsafe_allow_html=True)
    generate=st.button("⚡  Generate next-hour forecast",use_container_width=True)

selected_dt=pd.to_datetime(f"{chosen_date} {chosen_time}")
found=df.index[df.datetime==selected_dt].tolist()
if not found: st.error("Selected timestamp is not available."); st.stop()
idx=found[0]
if idx<23: st.warning("Choose a later hour; the model needs 24 preceding hourly records."); st.stop()

if "forecast_history" not in st.session_state: st.session_state.forecast_history=[]
if generate:
    try:
        x=df.loc[idx-23:idx,FEATURES].values
        x=feature_scaler.transform(x).reshape(1,24,len(FEATURES))
        y=model.predict(x,verbose=0)
        pred=max(0.0,float(target_scaler.inverse_transform(np.asarray(y).reshape(-1,1))[0,0]))
        next_time=selected_dt+timedelta(hours=1)
        item={"reference":selected_dt.strftime("%d %b %Y · %H:%M"),"time":next_time.strftime("%d %b %Y · %H:%M"),"value":pred}
        st.session_state["latest_forecast"]=item
        st.session_state.forecast_history.insert(0,item)
        st.session_state.forecast_history=st.session_state.forecast_history[:10]
    except Exception as e: st.error(f"Prediction failed: {e}")

result=st.session_state.get("latest_forecast")
if result:
    st.markdown("### Forecast result")
    x1,x2=st.columns([1.25,1],gap="medium")
    with x1:
        st.markdown('<div class="card result"><div class="kicker">PREDICTED RENEWABLE GENERATION · NEXT HOUR</div><div class="big">'+f'{result["value"]:,.2f}'+'</div><div class="sub">Forecast time: '+result["time"]+'</div><div class="sub">Reported in the dataset original unit</div></div>',unsafe_allow_html=True)
    with x2:
        st.markdown('<div class="card"><div class="kicker">ILLUSTRATIVE EV CHARGING EQUIVALENT</div><div class="big">'+f'{result["value"]/25:,.1f} EVs'+'</div><div class="sub">Assumes 25 kWh per EV · illustrative comparison only</div></div>',unsafe_allow_html=True)
    st.markdown("")
    st.markdown("#### Recent generation and forecast point")
    plot=df.loc[max(0,idx-23):idx,["datetime","Total_Renewable"]].copy()
    plot=plot.rename(columns={"datetime":"Time","Total_Renewable":"Observed"})
    future=pd.DataFrame({"Time":[selected_dt+timedelta(hours=1)],"Observed":[np.nan],"Forecast":[result["value"]]})
    plot["Forecast"]=np.nan
    chart=pd.concat([plot,future],ignore_index=True).set_index("Time")
    st.line_chart(chart,use_container_width=True)
    st.caption("Observed generation over the 24-hour input window; the final marker is the model's next-hour estimate.")
else:
    st.info("Select a date and reference hour above, then press Generate. Your forecast and supporting charts will appear here.")

st.divider()
st.markdown("### Generation history")
st.caption("Forecasts generated during this browser session, newest first.")
if st.session_state.forecast_history:
    h=pd.DataFrame(st.session_state.forecast_history)
    h=h.rename(columns={"reference":"Input reference","time":"Predicted hour","value":"Predicted generation"})
    h["Predicted generation"]=h["Predicted generation"].map(lambda v:f"{v:,.2f}")
    st.dataframe(h,use_container_width=True,hide_index=True)
else: st.caption("No forecasts generated in this session yet.")

with st.expander("Explore weather indicators (solar and wind resource)"):
    st.write("The dataset contains radiation and wind-speed measurements, but does not provide separate solar-generation and wind-generation outputs. These charts show weather/resource indicators, not a solar-versus-wind generation split.")
    window=df.loc[idx-23:idx]
    l,r=st.columns(2)
    with l:
        st.markdown("**Solar resource · shortwave radiation**")
        st.line_chart(window.set_index("datetime")[["shortwave_radiation"]],use_container_width=True)
    with r:
        st.markdown("**Wind resource · wind speed at 10 m**")
        st.line_chart(window.set_index("datetime")[["windspeed_10m"]],use_container_width=True)

with st.expander("How the forecast works"):
    st.markdown("""
- **Input:** the selected reference hour and the preceding 23 hourly records (24 records total).
- **Model:** trained GRU network with the saved feature and target scalers.
- **Horizon:** one hour ahead.
- **EV equivalent:** an illustrative division by an assumed 25 kWh per EV; not a measured number of vehicles or a guarantee of charging capability.
- **Data:** historical dataset, not a live grid feed.
""")
st.markdown('<div class="smallnote">GreenGrid · GRU-based renewable generation forecasting prototype</div>',unsafe_allow_html=True)
