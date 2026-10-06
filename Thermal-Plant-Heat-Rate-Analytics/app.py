from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Heat Rate Analytics", page_icon="⚡", layout="wide")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sample_plant_data.csv"
MODEL = ROOT / "models" / "heat_rate_model.joblib"
METRICS = ROOT / "models" / "metrics.json"
FEATURES = ["load_mw","coal_gcv_kcal_kg","main_steam_temp_c","main_steam_pressure_mpa","reheat_temp_c","condenser_vacuum_bar","aux_power_pct","boiler_efficiency_pct","ambient_temp_c"]

st.title("⚡ Thermal Plant Heat Rate Analytics")
st.caption("Industrial data science portfolio demonstration using synthetic data")

if not MODEL.exists():
    st.warning("Model not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL)
df = pd.read_csv(DATA)
pred = model.predict(df[FEATURES])

c1,c2,c3,c4=st.columns(4)
c1.metric("Average Heat Rate", f"{df.gross_heat_rate_kcal_kwh.mean():,.0f} kcal/kWh")
c2.metric("Average Load", f"{df.load_mw.mean():,.0f} MW")
c3.metric("Boiler Efficiency", f"{df.boiler_efficiency_pct.mean():.1f}%")
c4.metric("Auxiliary Power", f"{df.aux_power_pct.mean():.1f}%")

left,right=st.columns(2)
with left:
    chart=pd.DataFrame({"Actual":df.gross_heat_rate_kcal_kwh,"Predicted":pred})
    st.plotly_chart(px.scatter(chart,x="Actual",y="Predicted",title="Actual vs Predicted Heat Rate",opacity=.55),use_container_width=True)
with right:
    imp=pd.DataFrame({"Feature":FEATURES,"Importance":model.named_steps["model"].feature_importances_}).sort_values("Importance")
    st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h",title="Model Feature Importance"),use_container_width=True)

st.subheader("Single Operating-Point Prediction")
values={}
cols=st.columns(3)
labels={"load_mw":"Load (MW)","coal_gcv_kcal_kg":"Coal GCV (kcal/kg)","main_steam_temp_c":"Main Steam Temp (°C)","main_steam_pressure_mpa":"Main Steam Pressure (MPa)","reheat_temp_c":"Reheat Temp (°C)","condenser_vacuum_bar":"Condenser Vacuum (bar)","aux_power_pct":"Auxiliary Power (%)","boiler_efficiency_pct":"Boiler Efficiency (%)","ambient_temp_c":"Ambient Temp (°C)"}
for i,f in enumerate(FEATURES):
    values[f]=cols[i%3].number_input(labels[f],value=float(df[f].median()),format="%.3f")
if st.button("Predict Heat Rate",type="primary"):
    result=float(model.predict(pd.DataFrame([values]))[0])
    st.success(f"Predicted Gross Heat Rate: {result:,.1f} kcal/kWh")
    st.info("Use this prediction as an analytical reference. Validate recommendations with plant engineering procedures.")

with st.expander("Dataset preview"):
    st.dataframe(df.head(100),use_container_width=True)
