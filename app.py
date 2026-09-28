import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import timedelta
import tensorflow as tf

st.set_page_config(page_title="GreenGrid | Energy Control", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp{background:#071512;color:#e8f3ed}
section[data-testid="stSidebar"]{background:#10241e;border-right:1px solid #244b3c}
h1,h2,h3,p,label{color:#e8f3ed}
.eyebrow{font-size:11px;letter-spacing:2.5px;color:#80c8a0;font-weight:700}
.hero{padding:28px 30px;background:linear-gradient(115deg,#153b2d,#0c211a);border:1px solid #2a6248;border-radius:18px;margin:4px 0 22px}
.hero h1{font-size:38px;line-height:1.15;margin:8px 0}.hero p{color:#b4cfc0;font-size:15px;margin:0}
.panel{background:#102820;border:1px solid #28513f;border-radius:15px;padding:18px 20px}
.kicker{font-size:12px;color:#a8c6b5}.big{font-size:31px;font-weight:750;color:#75e0a2;line-height:1.2;margin:8px 0}
.muted{font-size:12px;color:#a8c6b5}
.stButton>button{background:#6bd99a;color:#082015;border:0;border-radius:10px;font-weight:700;min-height:44px}
.stButton>button:hover{background:#8ce8b1;color:#082015}
[data-testid="stMetric"]{background:#102820;padding:14px;border:1px solid #28513f;border-radius:12px}
[data-testid="stMetricLabel"]{color:#a8c6b5}
div[data-baseweb="tab-list"]{gap:8px}
button[data-baseweb="tab"]{background:#102820;border-radius:9px 9px 0 0;padding:10px 16px}
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
    return (tf.keras.models.load_model(MODEL_PATH,compile=False),
            joblib.load(FEATURE_SCALER_PATH),joblib.load(TARGET_SCALER_PATH))

st.markdown('<div class="hero"><div class="eyebrow">GREENGRID / ENERGY INTELLIGENCE</div><h1>Renewable Forecast Control Room</h1><p>Explore historical generation, select a forecast window, and run a one-hour-ahead GRU prediction.</p></div>',unsafe_allow_html=True)

missing=[p.name for p in [MODEL_PATH,FEATURE_SCALER_PATH,TARGET_SCALER_PATH] if not p.exists()]
if DATA_PATH is None: st.error("Dataset CSV not found in repository."); st.stop()
if missing: st.error("Missing model assets: "+", ".join(missing)); st.stop()
try:
    df=load_data(str(DATA_PATH)); model,feature_scaler,target_scaler=load_assets()
except Exception as e:
    st.error(f"Project files could not be loaded: {e}"); st.stop()
missing_cols=[c for c in FEATURES if c not in df]
if missing_cols: st.error("Dataset is missing required features: "+", ".join(missing_cols)); st.stop()
df[FEATURES]=df[FEATURES].apply(pd.to_numeric,errors="coerce")
df=df.dropna(subset=FEATURES).reset_index(drop=True)
if len(df)<25: st.error("Not enough valid hourly records."); st.stop()

# Overview strip
m1,m2,m3=st.columns(3)
m1.metric("Hourly records",f"{len(df):,}")
m2.metric("Dataset begins",df.datetime.min().strftime("%d %b %Y"))
m3.metric("Latest record",df.datetime.max().strftime("%d %b %Y"))
forecast_tab, explore_tab, method_tab=st.tabs(["⚡ Forecast workspace","▥ Data explorer","ⓘ Model notes"])

with st.sidebar:
    st.markdown("## Forecast setup")
    st.caption("STEP 1 · Choose a historical timestamp")
    min_date=df.datetime.min().date(); max_date=df.datetime.max().date()
    chosen_date=st.date_input("Dataset date",value=max_date,min_value=min_date,max_value=max_date)
    day=df[df.datetime.dt.date==chosen_date]
    if day.empty:
        st.warning("No records on this date."); st.stop()
    times=day.datetime.dt.strftime("%H:%M").tolist()
    chosen_time=st.selectbox("Reference hour",times,index=len(times)-1)
    st.caption("STEP 2 · Run the model")
    run=st.button("Generate next-hour forecast",use_container_width=True)
    st.divider()
    st.caption("Model input: 24 consecutive hourly records")
    st.caption("Output horizon: +1 hour")

selected_dt=pd.to_datetime(f"{chosen_date} {chosen_time}")
matches=df.index[df.datetime==selected_dt].tolist()
if not matches: st.error("Timestamp not found."); st.stop()
idx=matches[0]
if idx<23: st.warning("Select a later timestamp so 24 preceding hourly records are available."); st.stop()

with forecast_tab:
    left,right=st.columns([1.2,1],gap="large")
    with left:
        st.markdown("### Forecast workspace")
        st.write("The selected reference hour is **"+selected_dt.strftime("%d %B %Y · %H:%M")+"**.")
        st.markdown('<div class="panel"><div class="kicker">INPUT WINDOW</div><div class="big">24 hours</div><div class="muted">Recent weather and renewable-generation features are passed to the trained GRU.</div></div>',unsafe_allow_html=True)
        st.markdown("#### Recent generation profile")
        hist=df.loc[idx-23:idx,["datetime","Total_Renewable"]].set_index("datetime")
        st.area_chart(hist, use_container_width=True)
    with right:
        st.markdown("### Forecast result")
        if run:
            try:
                x=df.loc[idx-23:idx,FEATURES].values
                x=feature_scaler.transform(x).reshape(1,24,len(FEATURES))
                y=model.predict(x,verbose=0)
                prediction=max(0.0,float(target_scaler.inverse_transform(np.asarray(y).reshape(-1,1))[0,0]))
                forecast_time=selected_dt+timedelta(hours=1)
                st.session_state["forecast_result"]={"value":prediction,"time":forecast_time}
            except Exception as e: st.error(f"Prediction failed: {e}")
        result=st.session_state.get("forecast_result")
        if result:
            st.markdown('<div class="panel"><div class="kicker">PREDICTED RENEWABLE GENERATION</div><div class="big">'+f'{result["value"]:,.2f}'+'</div><div class="muted">Dataset original unit · '+result["time"].strftime("%d %b %Y, %H:%M")+'</div></div>',unsafe_allow_html=True)
            st.markdown("")
            st.metric("Illustrative EV equivalent",f'{result["value"]/25:,.1f} EVs')
            st.caption("Illustration only: assumes 25 kWh per EV. It is not a measured vehicle count.")
        else:
            st.info("Choose a timestamp in the sidebar, then press **Generate next-hour forecast**. Your result will appear here.")

with explore_tab:
    st.markdown("### Historical data explorer")
    st.write("Inspect the dataset before running a forecast.")
    start,end=st.columns(2)
    with start: range_start=st.date_input("From",value=max(min_date,max_date-timedelta(days=6)),min_value=min_date,max_value=max_date,key="range_start")
    with end: range_end=st.date_input("To",value=max_date,min_value=min_date,max_value=max_date,key="range_end")
    if range_start>range_end: st.warning("Start date must be before end date.")
    else:
        view=df[(df.datetime.dt.date>=range_start)&(df.datetime.dt.date<=range_end)]
        st.metric("Records in selected range",f"{len(view):,}")
        if not view.empty:
            st.line_chart(view.set_index("datetime")[["Total_Renewable"]],use_container_width=True)
            st.dataframe(view[["datetime","Total_Renewable","temperature_2m","cloudcover","windspeed_10m"]].tail(100),use_container_width=True,hide_index=True)

with method_tab:
    st.markdown("### How this prototype works")
    st.markdown("""
**1 · Historical input** — the selected timestamp and its 23 previous hourly records form a 24-hour input window.

**2 · Preprocessing** — the saved feature scaler transforms the input using the training-time scaling.

**3 · GRU inference** — the trained model estimates total renewable generation one hour ahead.

**4 · Output** — the saved target scaler converts the prediction back to the dataset's original unit.

**EV comparison** — shown only as an illustrative equivalent using an assumed 25 kWh per EV. Actual charging capacity cannot be inferred without confirmed energy units and system constraints.
""")
    st.info("This is a forecasting prototype using historical dataset records; it is not a live grid feed.")
st.divider()
st.caption("GreenGrid · GRU-based one-hour renewable generation forecasting prototype")
