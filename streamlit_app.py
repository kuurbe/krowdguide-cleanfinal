# streamlit_app.py
import os
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import humanize
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ------------------ App Config ------------------
st.set_page_config(page_title="KrowdGuide Smart City OS", layout="wide")
st.title("🚗 KrowdGuide — Deep Ellum Intelligence Hub")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIRS = [
    BASE_DIR / "data",
    Path(r"C:\Users\juhco\OneDrive\Documents\krowdguide-cleanfinal\data"),
]

def first_existing_dir(paths):
    for p in paths:
        if p and p.exists():
            return p
    return None

DATA_DIR = first_existing_dir(DATA_DIRS)
if DATA_DIR is None:
    st.error("❌ No `data/` folder found.")
    st.stop()

# ------------------ Helpers ------------------
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    try:
        if str(path).endswith(".gz"):
            return pd.read_csv(path, compression="infer", low_memory=False, on_bad_lines="skip")
        return pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, low_memory=False, on_bad_lines="skip", engine="python")

def detect_col(df: pd.DataFrame, *keys):
    cols = [str(c).lower() for c in df.columns]
    for key in keys:
        k = key.lower()
        for i, c in enumerate(cols):
            if k in c or c.startswith(k):
                return df.columns[i]
    return None

def predict_series(dates, values, days_ahead=30):
    if len(dates) < 2:
        return None, None
    df = pd.DataFrame({"date": pd.to_datetime(dates), "value": values})
    df = df.dropna().sort_values("date").drop_duplicates("date")
    if len(df) < 2:
        return None, None
    df["days"] = (df["date"] - df["date"].min()).dt.days
    X = df[["days"]].values
    y = df["value"].values
    model = make_pipeline(PolynomialFeatures(degree=min(3, len(df)-1)), LinearRegression())
    model.fit(X, y)
    last_day = df["days"].max()
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1).reshape(-1, 1)
    future_dates = [df["date"].max() + timedelta(days=i) for i in range(1, days_ahead+1)]
    preds = model.predict(future_days)
    return future_dates, preds

# ------------------ Load Data ------------------
CSV_PATHS = sorted(list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.csv.gz")))
if not CSV_PATHS:
    st.warning("⚠️ No CSV files in `data/`")
    st.stop()

def find_file(keyword):
    for p in CSV_PATHS:
        if keyword.lower() in p.name.lower():
            return p
    return None

# Load all datasets
datasets = {
    "visits": load_csv(find_file("DeepEllumVisits") or find_file("WeeklyVisits")) if find_file("DeepEllumVisits") or find_file("WeeklyVisits") else pd.DataFrame(),
    "bike_ped": load_csv(find_file("bike_pedestrian") or find_file("bike") or find_file("pedestrian")) if find_file("bike_pedestrian") or find_file("bike") or find_file("pedestrian") else pd.DataFrame(),
    "txdot": load_csv(find_file("TxDOT")) if find_file("TxDOT") else pd.DataFrame(),
    "weather": load_csv(find_file("DeepWeather")) if find_file("DeepWeather") else pd.DataFrame(),
    "service": load_csv(find_file("311") or find_file("service")) if find_file("311") or find_file("service") else pd.DataFrame(),
    "arrests": load_csv(find_file("arrests")) if find_file("arrests") else pd.DataFrame(),
}

# ------------------ Uber/Lyft Style Dashboard ------------------
st.subheader("📍 Deep Ellum Live Intelligence")

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    vcol = detect_col(datasets["visits"], "visits", "count")
    last_visits = int(datasets["visits"][vcol].iloc[-1]) if not datasets["visits"].empty and vcol else 0
    st.metric("Foot Traffic", f"{last_visits:,}")
with col2:
    st.metric("Bike/Ped", f"{len(datasets['bike_ped']):,}")
with col3:
    st.metric("311 Requests", f"{len(datasets['service']):,}")
with col4:
    st.metric("Arrests", f"{len(datasets['arrests']):,}")

# Main visualization area
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Foot Traffic", "Bike/Ped", "TxDOT", "Weather", "Public Safety", "311"])

# Foot Traffic
with tab1:
    if not datasets["visits"].empty:
        wk = detect_col(datasets["visits"], "week", "date")
        vcol = detect_col(datasets["visits"], "visits")
        venue = detect_col(datasets["visits"], "venue", "location")
        if wk and vcol:
            fig = px.line(datasets["visits"].head(50), x=wk, y=vcol, color=venue, title="Foot Traffic (Last 50 Records)", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(datasets["visits"].head(50), use_container_width=True)

# Bike/Ped
with tab2:
    if not datasets["bike_ped"].empty:
        date_col = detect_col(datasets["bike_ped"], "date")
        count_col = detect_col(datasets["bike_ped"], "count", "volume")
        loc_col = detect_col(datasets["bike_ped"], "location", "area")
        if date_col and count_col:
            fig = px.line(datasets["bike_ped"].head(50), x=date_col, y=count_col, color=loc_col, title="Bike/Ped (Last 50 Records)", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(datasets["bike_ped"].head(50), use_container_width=True)

# TxDOT
with tab3:
    if not datasets["txdot"].empty:
        date_col = detect_col(datasets["txdot"], "date")
        count_col = detect_col(datasets["txdot"], "count", "incidents")
        loc_col = detect_col(datasets["txdot"], "location")
        if date_col and count_col:
            fig = px.line(datasets["txdot"].head(50), x=date_col, y=count_col, color=loc_col, title="TxDOT Incidents (Last 50 Records)", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(datasets["txdot"].head(50), use_container_width=True)

# Weather
with tab4:
    if not datasets["weather"].empty:
        date_col = detect_col(datasets["weather"], "date")
        temp_col = detect_col(datasets["weather"], "temp", "temperature")
        if date_col and temp_col:
            df = datasets["weather"][[date_col, temp_col]].dropna().head(50)
            df[date_col] = pd.to_datetime(df[date_col])
            future_dates, preds = predict_series(df[date_col], df[temp_col])
            if future_dates is not None:
                pred_df = pd.DataFrame({date_col: future_dates, temp_col: preds, "type": "Predicted"})
                actual_df = df.copy()
                actual_df["type"] = "Actual"
                full_df = pd.concat([actual_df, pred_df])
                fig = px.line(full_df, x=date_col, y=temp_col, color="type", title="Weather Forecast (Last 50 + Predicted)")
                st.plotly_chart(fig, use_container_width=True)
        st.dataframe(datasets["weather"].head(50), use_container_width=True)

# Public Safety
with tab5:
    if not datasets["arrests"].empty:
        cat_col = detect_col(datasets["arrests"], "offense", "category")
        loc_col = detect_col(datasets["arrests"], "location", "area")
        if cat_col:
            top_offenses = datasets["arrests"][cat_col].value_counts().head(10)
            fig = px.bar(x=top_offenses.values, y=top_offenses.index, orientation='h', title="Top Offenses")
            st.plotly_chart(fig, use_container_width=True)
        if loc_col:
            loc_counts = datasets["arrests"][loc_col].value_counts().head(10)
            fig = px.bar(x=loc_counts.values, y=loc_counts.index, orientation='h', title="Arrests by Location")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(datasets["arrests"].head(50), use_container_width=True)

# 311 Requests
with tab6:
    if not datasets["service"].empty:
        req_col = detect_col(datasets["service"], "request", "topic")
        loc_col = detect_col(datasets["service"], "location")
        if req_col:
            top_reqs = datasets["service"][req_col].value_counts().head(10)
            fig = px.bar(x=top_reqs.values, y=top_reqs.index, orientation='h', title="Top 311 Requests")
            st.plotly_chart(fig, use_container_width=True)
        if loc_col:
            loc_reqs = datasets["service"][loc_col].value_counts().head(10)
            fig = px.bar(x=loc_reqs.values, y=loc_reqs.index, orientation='h', title="311 by Location")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(datasets["service"].head(50), use_container_width=True)

# Footer
st.caption(f"📍 Deep Ellum Intelligence Hub · Data folder: `{DATA_DIR}` · Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")