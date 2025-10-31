# streamlit_app.py
import os
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import humanize
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ------------------ App Config ------------------
st.set_page_config(
    page_title="KrowdGuide — Deep Ellum Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .golden-ticket {
        background: linear-gradient(135deg, #f0c850, #d4af37);
        color: white;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
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
    "weather": load_csv(find_file("DeepWeather") or find_file("weather") or find_file("DeepEllumWeather")) if find_file("DeepWeather") or find_file("weather") or find_file("DeepEllumWeather") else pd.DataFrame(),
    "service": load_csv(find_file("311") or find_file("service")) if find_file("311") or find_file("service") else pd.DataFrame(),
    "arrests": load_csv(find_file("arrests") or find_file("crime") or find_file("police")) if find_file("arrests") or find_file("crime") or find_file("police") else pd.DataFrame(),
}

# ------------------ Sidebar Filters ------------------
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Select Date Range", value=[datetime.now() - timedelta(days=365), datetime.now()])

# Process datasets based on filters
filtered_datasets = {}
for key, df in datasets.items():
    if not df.empty:
        date_col = detect_col(df, "date", "datetime", "time")
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            start_date, end_date = date_range
            filtered_datasets[key] = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
        else:
            filtered_datasets[key] = df

# ------------------ Anonymize Arrests Data ------------------
if not filtered_datasets["arrests"].empty:
    # Identify sensitive columns
    sensitive_cols = []
    for col in filtered_datasets["arrests"].columns:
        col_lower = col.lower()
        if any(sensitive in col_lower for sensitive in ["name", "race", "ethnicity", "address", "officer"]):
            sensitive_cols.append(col)
    
    # Remove sensitive columns
    for col in sensitive_cols:
        if col in filtered_datasets["arrests"].columns:
            filtered_datasets["arrests"] = filtered_datasets["arrests"].drop(columns=[col])

# ------------------ Executive Dashboard ------------------
# Golden Ticket Header
st.markdown('<div class="golden-ticket"><h2>🌟 Deep Ellum: The Golden Ticket for Urban Investment</h2><p>Real-time intelligence for business growth, safety, and city planning</p></div>', unsafe_allow_html=True)

# KPIs in polished cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    vcol = detect_col(filtered_datasets["visits"], "visits", "count")
    venue_col = detect_col(filtered_datasets["visits"], "venue", "business", "location")
    last_visits = int(filtered_datasets["visits"][vcol].iloc[-1]) if not filtered_datasets["visits"].empty and vcol else 0
    unique_businesses = filtered_datasets["visits"][venue_col].nunique() if venue_col and not filtered_datasets["visits"].empty else 0
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
        <div class="metric-value">{len(filtered_datasets['bike_ped']):,}</div>
        <div class="metric-label">Bike/Ped Records</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(filtered_datasets['service']):,}</div>
        <div class="metric-label">311 Requests</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(filtered_datasets['arrests']):,}</div>
        <div class="metric-label">Public Safety Incidents</div>
    </div>
    """, unsafe_allow_html=True)

# Automated Alerts
if not filtered_datasets["arrests"].empty:
    arrest_date_col = detect_col(filtered_datasets["arrests"], "date", "datetime")
    arrest_loc_col = detect_col(filtered_datasets["arrests"], "location", "area", "address")
    arrest_cat_col = detect_col(filtered_datasets["arrests"], "offense", "crime", "category")
    
    if arrest_date_col:
        recent_arrests = filtered_datasets["arrests"][filtered_datasets["arrests"][arrest_date_col] >= (datetime.now() - timedelta(days=7))]
        if len(recent_arrests) > 0:
            st.markdown(f"""
            <div class="alert-box">
                ⚠️ <strong>ALERT:</strong> {len(recent_arrests)} arrests recorded in Deep Ellum in the last 7 days. 
                <a href="#" target="_blank">View Details</a>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div class="insight-box">💡 <strong>Insight:</strong> Real-time urban analytics for Deep Ellum — enabling data-driven decisions for businesses, city planners, and investors.</div>', unsafe_allow_html=True)

# Tabs for interactive views
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Historical Data", "🚗 Traffic", "🚶 Foot Traffic", "🚴 Bike/Ped", "👮 Safety", "311", "🔮 Predictions"])

# Historical Data View
with tab1:
    st.subheader("Historical Data Overview")
    combined_df = pd.DataFrame()
    
    for key, df in filtered_datasets.items():
        if not df.empty:
            date_col = detect_col(df, "date", "datetime", "time")
            if date_col:
                df_subset = df[[date_col]].copy()
                df_subset['type'] = key
                df_subset['count'] = len(df)
                combined_df = pd.concat([df_subset, combined_df], ignore_index=True)
    
    if not combined_df.empty:
        fig = px.line(combined_df, x=date_col, y='count', color='type', title="Historical Data Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed historical data table
        st.subheader("Detailed Historical Data")
        for key, df in filtered_datasets.items():
            if not df.empty:
                st.markdown(f"### {key.replace('_', ' ').title()}")
                st.dataframe(df, use_container_width=True)
    else:
        st.info("No data available for the selected date range.")

# Traffic View
with tab2:
    st.subheader("Traffic Data")
    if not filtered_datasets["txdot"].empty:
        date_col = detect_col(filtered_datasets["txdot"], "date")
        count_col = detect_col(filtered_datasets["txdot"], "incidents")
        loc_col = detect_col(filtered_datasets["txdot"], "location", "area")
        if date_col and count_col:
            fig = px.line(filtered_datasets["txdot"], x=date_col, y=count_col, title="Traffic Incidents", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Traffic Heatmap
        if loc_col:
            loc_counts = filtered_datasets["txdot"][loc_col].value_counts()
            heatmap_df = pd.DataFrame({
                'Location': loc_counts.index,
                'Incident Count': loc_counts.values
            })
            fig2 = px.density_heatmap(
                data_frame=heatmap_df,
                x='Location',
                y='Incident Count',
                title="Traffic Incidents Heatmap by Location",
                labels=dict(x="Location", y="Incident Count")
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        st.dataframe(filtered_datasets["txdot"], use_container_width=True)
    else:
        st.info("No traffic data available.")

# Foot Traffic View
with tab3:
    st.subheader("Foot Traffic Data")
    if not filtered_datasets["visits"].empty:
        wk = detect_col(filtered_datasets["visits"], "week", "date")
        vcol = detect_col(filtered_datasets["visits"], "visits")
        venue_col = detect_col(filtered_datasets["visits"], "venue", "business", "location")
        if wk and vcol:
            if venue_col:
                fig = px.line(filtered_datasets["visits"], x=wk, y=vcol, color=venue_col, title="Foot Traffic by Business", markers=True)
            else:
                fig = px.line(filtered_datasets["visits"], x=wk, y=vcol, title="Foot Traffic", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Foot Traffic Heatmap
        if venue_col:
            venue_counts = filtered_datasets["visits"][venue_col].value_counts()
            heatmap_df = pd.DataFrame({
                'Business': venue_counts.index,
                'Visitor Count': venue_counts.values
            })
            fig2 = px.density_heatmap(
                data_frame=heatmap_df,
                x='Business',
                y='Visitor Count',
                title="Foot Traffic Heatmap by Business",
                labels=dict(x="Business", y="Visitor Count")
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        st.dataframe(filtered_datasets["visits"], use_container_width=True)
    else:
        st.info("No foot traffic data available.")

# Bike/Ped View
with tab4:
    st.subheader("Bike/Ped Data")
    if not filtered_datasets["bike_ped"].empty:
        date_col = detect_col(filtered_datasets["bike_ped"], "date")
        count_col = detect_col(filtered_datasets["bike_ped"], "count")
        loc_col = detect_col(filtered_datasets["bike_ped"], "location")
        if date_col and count_col:
            if loc_col:
                fig = px.line(filtered_datasets["bike_ped"], x=date_col, y=count_col, color=loc_col, title="Bike/Ped Activity by Location", markers=True)
            else:
                fig = px.line(filtered_datasets["bike_ped"], x=date_col, y=count_col, title="Bike/Ped Activity", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Bike/Ped Heatmap
        if loc_col:
            loc_counts = filtered_datasets["bike_ped"][loc_col].value_counts()
            heatmap_df = pd.DataFrame({
                'Location': loc_counts.index,
                'Activity Count': loc_counts.values
            })
            fig2 = px.density_heatmap(
                data_frame=heatmap_df,
                x='Location',
                y='Activity Count',
                title="Bike/Ped Activity Heatmap by Location",
                labels=dict(x="Location", y="Activity Count")
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        st.dataframe(filtered_datasets["bike_ped"], use_container_width=True)
    else:
        st.info("No bike/ped data available.")

# Safety View
with tab5:
    st.subheader("Safety Data (Arrests & Crimes)")
    if not filtered_datasets["arrests"].empty:
        cat_col = detect_col(filtered_datasets["arrests"], "offense", "crime", "category")
        loc_col = detect_col(filtered_datasets["arrests"], "location", "area", "address")
        
        if cat_col:
            top_offenses = filtered_datasets["arrests"][cat_col].value_counts().head(10)
            fig = px.bar(x=top_offenses.values, y=top_offenses.index, orientation='h', title="Top Offense Categories")
            st.plotly_chart(fig, use_container_width=True)
        
        if loc_col:
            loc_counts = filtered_datasets["arrests"][loc_col].value_counts().head(10)
            fig2 = px.bar(x=loc_counts.values, y=loc_counts.index, orientation='h', title="Arrests by Location")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Safety Heatmap
        if loc_col:
            loc_counts = filtered_datasets["arrests"][loc_col].value_counts()
            heatmap_df = pd.DataFrame({
                'Location': loc_counts.index,
                'Incident Count': loc_counts.values
            })
            fig3 = px.density_heatmap(
                data_frame=heatmap_df,
                x='Location',
                y='Incident Count',
                title="Crime Heatmap by Location",
                labels=dict(x="Location", y="Incident Count")
            )
            st.plotly_chart(fig3, use_container_width=True)
            
        st.dataframe(filtered_datasets["arrests"], use_container_width=True)
    else:
        st.info("No arrest data available.")

# 311 View
with tab6:
    st.subheader("311 Requests Data")
    if not filtered_datasets["service"].empty:
        req_col = detect_col(filtered_datasets["service"], "request", "topic")
        loc_col = detect_col(filtered_datasets["service"], "location")
        
        if req_col:
            top_reqs = filtered_datasets["service"][req_col].value_counts().head(10)
            fig3 = px.pie(values=top_reqs.values, names=top_reqs.index, title="311 Request Distribution")
            st.plotly_chart(fig3, use_container_width=True)
        
        if loc_col:
            loc_reqs = filtered_datasets["service"][loc_col].value_counts().head(10)
            fig4 = px.bar(x=loc_reqs.values, y=loc_reqs.index, orientation='h', title="311 Requests by Location")
            st.plotly_chart(fig4, use_container_width=True)
        
        # 311 Heatmap
        if loc_col:
            loc_counts = filtered_datasets["service"][loc_col].value_counts()
            heatmap_df = pd.DataFrame({
                'Location': loc_counts.index,
                'Request Count': loc_counts.values
            })
            fig5 = px.density_heatmap(
                data_frame=heatmap_df,
                x='Location',
                y='Request Count',
                title="311 Requests Heatmap by Location",
                labels=dict(x="Location", y="Request Count")
            )
            st.plotly_chart(fig5, use_container_width=True)
            
        st.dataframe(filtered_datasets["service"], use_container_width=True)
    else:
        st.info("No 311 request data available.")

# Predictions View
with tab7:
    st.subheader("Predictive Analytics")
    if not filtered_datasets["weather"].empty:
        date_col = detect_col(filtered_datasets["weather"], "time", "date", "datetime")
        temp_col = detect_col(filtered_datasets["weather"], "temp", "temperature", "temperature_2m_max")
        
        if date_col and temp_col:
            df = filtered_datasets["weather"][[date_col, temp_col]].dropna().head(50)
            df[date_col] = pd.to_datetime(df[date_col])
            future_dates, preds = predict_series(df[date_col], df[temp_col])
            if future_dates is not None:
                pred_df = pd.DataFrame({date_col: future_dates, temp_col: preds, "type": "Predicted"})
                actual_df = df.copy()
                actual_df["type"] = "Historical"
                
                full_df = pd.concat([actual_df, pred_df])
                fig = px.line(full_df, x=date_col, y=temp_col, color="type", title="Weather Prediction: Historical + Forecast")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction table
                st.subheader("Weather Prediction Table")
                st.dataframe(pred_df, use_container_width=True)
        else:
            st.info("No weather data available for predictions.")
    else:
        st.info("No weather data available for predictions.")

# Footer
st.divider()
st.caption("KrowdGuide — Transforming Urban Data into Strategic Intelligence | Deep Ellum Focus")