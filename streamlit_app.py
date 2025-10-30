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
st.set_page_config(
    page_title="KrowdGuide — Deep Ellum Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for investor-grade polish
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
    }
    .insight-box {
        background: #f8fafc;
        border-left: 4px solid #3182ce;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
    }
    .alert-box {
        background: #fff5f5;
        border-left: 4px solid #e53e3e;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
    }
    .safe-box {
        background: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
    }
    .new-age-container {
        background: linear-gradient(135deg, #f0f9ff, #e6f7ff);
        padding: 20px;
        border-radius: 16px;
        margin: 20px 0;
    }
    .data-section {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">KrowdGuide</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Ellum Intelligence Dashboard — Investor View</div>', unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

if not DATA_DIR.exists():
    st.error("❌ Data folder not found. Please ensure `data/` exists with your CSVs.")
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
    if df.empty:
        return None
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
    st.warning("⚠️ No datasets found in `data/` folder.")
    st.stop()

def find_file(keyword):
    for p in CSV_PATHS:
        if keyword.lower() in p.name.lower():
            return p
    return None

datasets = {
    "visits": load_csv(find_file("DeepEllumVisits") or find_file("WeeklyVisits")) if find_file("DeepEllumVisits") or find_file("WeeklyVisits") else pd.DataFrame(),
    "bike_ped": load_csv(find_file("bike_pedestrian") or find_file("bike") or find_file("pedestrian")) if find_file("bike_pedestrian") or find_file("bike") or find_file("pedestrian") else pd.DataFrame(),
    "txdot": load_csv(find_file("TxDOT")) if find_file("TxDOT") else pd.DataFrame(),
    "weather": load_csv(find_file("DeepWeather") or find_file("weather")) if find_file("DeepWeather") or find_file("weather") else pd.DataFrame(),
    "service": load_csv(find_file("311") or find_file("service")) if find_file("311") or find_file("service") else pd.DataFrame(),
    "arrests": load_csv(find_file("arrests") or find_file("crime") or find_file("police")) if find_file("arrests") or find_file("crime") or find_file("police") else pd.DataFrame(),
}

# ------------------ Executive Dashboard ------------------
# KPIs in polished cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    vcol = detect_col(datasets["visits"], "visits", "count")
    venue_col = detect_col(datasets["visits"], "venue", "business", "location")
    last_visits = int(datasets["visits"][vcol].iloc[-1]) if not datasets["visits"].empty and vcol else 0
    unique_businesses = datasets["visits"][venue_col].nunique() if venue_col and not datasets["visits"].empty else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{last_visits:,}</div>
        <div class="metric-label">Weekly Foot Traffic</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{unique_businesses:,}</div>
        <div class="metric-label">Tracked Businesses</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(datasets['bike_ped']):,}</div>
        <div class="metric-label">Bike/Ped Records</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(datasets['service']):,}</div>
        <div class="metric-label">311 Requests</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(datasets['arrests']):,}</div>
        <div class="metric-label">Public Safety Incidents</div>
    </div>
    """, unsafe_allow_html=True)

# Automated Alerts
if not datasets["arrests"].empty:
    arrest_date_col = detect_col(datasets["arrests"], "date", "datetime")
    arrest_loc_col = detect_col(datasets["arrests"], "location", "area", "address")
    arrest_cat_col = detect_col(datasets["arrests"], "offense", "crime", "category")
    
    if arrest_date_col:
        recent_arrests = datasets["arrests"][datasets["arrests"][arrest_date_col] >= (datetime.now() - timedelta(days=7))]
        if len(recent_arrests) > 0:
            st.markdown(f"""
            <div class="alert-box">
                ⚠️ <strong>ALERT:</strong> {len(recent_arrests)} arrests recorded in Deep Ellum in the last 7 days. 
                <a href="#" target="_blank">View Details</a>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div class="insight-box">💡 <strong>Insight:</strong> Real-time urban analytics for Deep Ellum — enabling data-driven decisions for businesses, city planners, and investors.</div>', unsafe_allow_html=True)

# Simplified Single View Dashboard
st.subheader("📊 Deep Ellum Comprehensive Intelligence View")

# Weather Trends
st.markdown("### 🌦️ Weather Trends")
if not datasets["weather"].empty:
    date_col = detect_col(datasets["weather"], "date")
    temp_col = detect_col(datasets["weather"], "temp", "temperature")
    precip_col = detect_col(datasets["weather"], "precip", "rain")
    hum_col = detect_col(datasets["weather"], "humidity")
    
    if date_col and temp_col:
        df = datasets["weather"][[date_col, temp_col]].dropna().head(50)
        df[date_col] = pd.to_datetime(df[date_col])
        future_dates, preds = predict_series(df[date_col], df[temp_col])
        if future_dates is not None:
            pred_df = pd.DataFrame({date_col: future_dates, temp_col: preds, "type": "Forecast"})
            actual_df = df.copy()
            actual_df["type"] = "Historical"
            full_df = pd.concat([actual_df, pred_df])
            fig = px.line(full_df, x=date_col, y=temp_col, color="type", title="Temperature: Historical + 30-Day Forecast")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Weather data not available. Integrate with AccuWeather API for live trends.")

# Traffic and Foot Traffic
st.markdown("### 🚗 Traffic & Foot Traffic")
col_a, col_b = st.columns(2)
with col_a:
    if not datasets["txdot"].empty:
        date_col = detect_col(datasets["txdot"], "date")
        count_col = detect_col(datasets["txdot"], "incidents")
        if date_col and count_col:
            fig = px.line(datasets["txdot"].head(50), x=date_col, y=count_col, title="Traffic Incidents (Last 50 Records)", markers=True)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No traffic data available.")
        
with col_b:
    if not datasets["visits"].empty:
        wk = detect_col(datasets["visits"], "week", "date")
        vcol = detect_col(datasets["visits"], "visits")
        venue_col = detect_col(datasets["visits"], "venue", "business", "location")
        if wk and vcol:
            df_plot = datasets["visits"].head(50)
            if venue_col:
                fig = px.line(df_plot, x=wk, y=vcol, color=venue_col, title="Foot Traffic by Business (Last 50 Records)", markers=True)
            else:
                fig = px.line(df_plot, x=wk, y=vcol, title="Foot Traffic (Last 50 Records)", markers=True)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No foot traffic data available.")

# Safety (Arrests & Crimes)
st.markdown("### 👮 Safety: Arrests & Crimes")
col_c, col_d = st.columns(2)
with col_c:
    if not datasets["arrests"].empty:
        cat_col = detect_col(datasets["arrests"], "offense", "crime", "category")
        if cat_col:
            top_offenses = datasets["arrests"][cat_col].value_counts().head(8)
            fig = px.bar(x=top_offenses.values, y=top_offenses.index, orientation='h', title="Top Offense Categories")
            fig.update_layout(height=300, margin=dict(l=120))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No arrest data available.")

with col_d:
    if not datasets["arrests"].empty:
        loc_col = detect_col(datasets["arrests"], "location", "area", "address")
        if loc_col:
            loc_counts = datasets["arrests"][loc_col].value_counts().head(8)
            fig2 = px.bar(x=loc_counts.values, y=loc_counts.index, orientation='h', title="Arrests by Location")
            fig2.update_layout(height=300, margin=dict(l=120))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No arrest location data available.")

# Bike/Ped & 311 Requests
st.markdown("### 🚴 Bike/Ped & 311 Requests")
col_e, col_f = st.columns(2)
with col_e:
    if not datasets["bike_ped"].empty:
        date_col = detect_col(datasets["bike_ped"], "date")
        count_col = detect_col(datasets["bike_ped"], "count")
        loc_col = detect_col(datasets["bike_ped"], "location")
        if date_col and count_col:
            if loc_col:
                fig = px.line(datasets["bike_ped"].head(50), x=date_col, y=count_col, color=loc_col, title="Bike/Ped Activity by Location", markers=True)
            else:
                fig = px.line(datasets["bike_ped"].head(50), x=date_col, y=count_col, title="Bike/Ped Activity", markers=True)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No bike/ped data available.")

with col_f:
    if not datasets["service"].empty:
        req_col = detect_col(datasets["service"], "request", "topic")
        loc_col = detect_col(datasets["service"], "location")
        if req_col:
            top_reqs = datasets["service"][req_col].value_counts().head(8)
            fig3 = px.pie(values=top_reqs.values, names=top_reqs.index, title="311 Request Distribution")
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No 311 request data available.")

# Raw Data Explorer
st.subheader("🔍 Raw Data Explorer (50 Records)")
for name, df in datasets.items():
    if not df.empty:
        with st.expander(f"📁 {name.replace('_', ' ').title()} ({len(df)} records)"):
            st.dataframe(df.head(50), use_container_width=True)

# Footer
st.divider()
st.caption("KrowdGuide — Transforming Urban Data into Strategic Intelligence | Deep Ellum Focus")