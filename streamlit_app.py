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
import requests

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
    /* Weather Widget Styling */
    .weather-widget {
        background: rgba(0, 0, 0, 0.7); /* Semi-transparent black background */
        color: white; /* White text */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ccc;
        max-width: 300px;
    }
    .weather-widget h4 {
        color: #ffffff;
        margin-top: 0;
    }
    .weather-widget p {
        margin: 5px 0;
    }
    /* Simplified Tab Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        background-color: #f0f0f0;
        border-radius: 8px 8px 0 0;
        padding: 0 16px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e0e0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3182ce;
        color: white;
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
    except Exception as e:
        st.warning(f"Warning: Could not load {path.name}. Error: {e}")
        return pd.DataFrame()

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
year_filter = st.sidebar.selectbox("Filter by Year", ["All", "2023", "2024", "2025"], index=0)

# Process datasets based on filters — ensure all keys exist
filtered_datasets = {}
for key, df in datasets.items():
    # Always include the key, even if empty
    if df.empty:
        filtered_datasets[key] = df
        continue

    date_col = detect_col(df, "date", "datetime", "time")
    if date_col:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        # Apply date range filter
        start_date, end_date = date_range
        df_filtered = df[(df[date_col] >= pd.Timestamp(start_date)) & (df[date_col] <= pd.Timestamp(end_date))]
        
        # Apply year filter if selected
        if year_filter != "All":
            df_filtered = df_filtered[df_filtered[date_col].dt.year == int(year_filter)]
        
        filtered_datasets[key] = df_filtered
    else:
        filtered_datasets[key] = df

# ------------------ Anonymize Arrests Data ------------------
if not filtered_datasets["arrests"].empty:
    # Identify sensitive columns
    sensitive_cols = []
    for col in filtered_datasets["arrests"].columns:
        col_lower = col.lower()
        if any(sensitive in col_lower for sensitive in ["name", "race", "ethnicity", "address", "officer", "person"]):
            sensitive_cols.append(col)
    
    # Replace sensitive data with placeholders
    for col in sensitive_cols:
        if col in filtered_datasets["arrests"].columns:
            # For name/address fields, replace with "Anonymous" or "Redacted"
            if col_lower in ["name", "officer", "person"]:
                filtered_datasets["arrests"][col] = "Anonymous"
            # For other sensitive data, replace with 0 or NaN if numeric, or "N/A" if string
            elif pd.api.types.is_numeric_dtype(filtered_datasets["arrests"][col]):
                filtered_datasets["arrests"][col] = 0
            else:
                filtered_datasets["arrests"][col] = "N/A"

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

# Insight Box with Black Text
st.markdown('<div class="insight-box"><span style="color: black;">💡 <strong>Insight:</strong> Real-time urban analytics for Deep Ellum — enabling data-driven decisions for businesses, city planners, and investors.</span></div>', unsafe_allow_html=True)

# Weather Widget (Python-based with enhanced details)
st.subheader("Current Dallas Weather")
try:
    # Use coordinates for Dallas
    latitude = 32.78
    longitude = -96.80
    api_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true&daily=temperature_2m_max,temperature_2m_min,precipitation_probability,weathercode,sunrise,sunset&timezone=America/Chicago"

    response = requests.get(api_url, timeout=10)
    response.raise_for_status()
    data = response.json()
    current = data.get('current_weather', {})
    daily = data.get('daily', {})
    
    temp = current.get('temperature', 'N/A')
    wind_speed = current.get('windspeed', 'N/A')
    wind_dir = current.get('winddirection', 'N/A')
    condition = current.get('weathercode', 'N/A')
    
    # Map weather codes to descriptions (simplified)
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    condition_desc = weather_codes.get(condition, f"Code {condition}")
    
    # Get today's forecast
    today_max = daily.get('temperature_2m_max', [])[0] if daily.get('temperature_2m_max') else 'N/A'
    today_min = daily.get('temperature_2m_min', [])[0] if daily.get('temperature_2m_min') else 'N/A'
    sunrise = daily.get('sunrise', [])[0] if daily.get('sunrise') else 'N/A'
    sunset = daily.get('sunset', [])[0] if daily.get('sunset') else 'N/A'
    humidity = daily.get('relative_humidity_2m', [])[0] if daily.get('relative_humidity_2m') else 'N/A'
    precipitation_prob = daily.get('precipitation_probability', [])[0] if daily.get('precipitation_probability') else 'N/A'

    st.markdown(f"""
    <div class="weather-widget">
        <h4>🌤️ Current Conditions</h4>
        <p><strong>Temperature:</strong> {temp}°C</p>
        <p><strong>Condition:</strong> {condition_desc}</p>
        <p><strong>Wind:</strong> {wind_speed} m/s from {wind_dir}°</p>
        <p><strong>Today's Forecast:</strong> High {today_max}°C, Low {today_min}°C</p>
        <p><strong>Sunrise/Sunset:</strong> {sunrise} / {sunset}</p>
        <p><strong>Humidity:</strong> {humidity}%</p>
        <p><strong>Precipitation Prob.:</strong> {precipitation_prob}%</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error("Unable to fetch weather data. Please check your internet connection.")

# Tabs for interactive views (Removed Predictions Tab)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚗 Traffic", "🚶 Foot Traffic", "🚴 Bike/Ped", "👮 Safety", "311", "🗺️ Map"])

# Traffic View
with tab1:
    st.subheader("Traffic Data")
    if not filtered_datasets["txdot"].empty:
        date_col = detect_col(filtered_datasets["txdot"], "date")
        count_col = detect_col(filtered_datasets["txdot"], "incidents")
        loc_col = detect_col(filtered_datasets["txdot"], "location", "area")
        if date_col and count_col:
            fig = px.line(filtered_datasets["txdot"], x=date_col, y=count_col, title="Traffic Incidents", markers=True)
            st.plotly_chart(fig, width="stretch")
        
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
            st.plotly_chart(fig2, width="stretch")
            
        st.dataframe(filtered_datasets["txdot"], width="stretch")
    else:
        st.info("No traffic data available.")

# Foot Traffic View
with tab2:
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
            st.plotly_chart(fig, width="stretch")
        
        # Pie Chart for Visitor Distribution (Accurate Percentages)
        if venue_col:
            venue_counts = filtered_datasets["visits"][venue_col].value_counts()
            pie_df = pd.DataFrame({
                'Business': venue_counts.index,
                'Visitor Count': venue_counts.values,
                'Percentage': (venue_counts.values / venue_counts.sum()) * 100  # Calculate percentage
            })
            # Format percentage to 1 decimal place
            pie_df['Percentage'] = pie_df['Percentage'].round(1)
            fig2 = px.pie(pie_df, values='Visitor Count', names='Business', title="Visitor Distribution by Business",
                         hover_data=['Percentage'], labels={'Percentage':'%'})
            fig2.update_traces(textinfo='percent+label', textfont_size=12)
            st.plotly_chart(fig2, width="stretch")
            
        st.dataframe(filtered_datasets["visits"], width="stretch")
    else:
        st.info("No foot traffic data available.")

# Bike/Ped View
with tab3:
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
            st.plotly_chart(fig, width="stretch")
        
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
            st.plotly_chart(fig2, width="stretch")
            
        st.dataframe(filtered_datasets["bike_ped"], width="stretch")
    else:
        st.info("No bike/ped data available.")

# Safety View
with tab4:
    st.subheader("Safety Data (Arrests & Crimes)")
    if not filtered_datasets["arrests"].empty:
        cat_col = detect_col(filtered_datasets["arrests"], "offense", "crime", "category")
        loc_col = detect_col(filtered_datasets["arrests"], "location", "area", "address")
        
        if cat_col:
            top_offenses = filtered_datasets["arrests"][cat_col].value_counts().head(10)
            fig = px.bar(x=top_offenses.values, y=top_offenses.index, orientation='h', title="Top Offense Categories")
            st.plotly_chart(fig, width="stretch")
        
        if loc_col:
            loc_counts = filtered_datasets["arrests"][loc_col].value_counts().head(10)
            fig2 = px.bar(x=loc_counts.values, y=loc_counts.index, orientation='h', title="Arrests by Location")
            st.plotly_chart(fig2, width="stretch")
        
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
            st.plotly_chart(fig3, width="stretch")
            
        st.dataframe(filtered_datasets["arrests"], width="stretch")
    else:
        st.info("No arrest data available.")

# 311 View
with tab5:
    st.subheader("311 Requests Data")
    if not filtered_datasets["service"].empty:
        req_col = detect_col(filtered_datasets["service"], "request", "topic")
        loc_col = detect_col(filtered_datasets["service"], "location")
        
        if req_col:
            top_reqs = filtered_datasets["service"][req_col].value_counts().head(10)
            fig3 = px.pie(values=top_reqs.values, names=top_reqs.index, title="311 Request Distribution")
            st.plotly_chart(fig3, width="stretch")
        
        if loc_col:
            loc_reqs = filtered_datasets["service"][loc_col].value_counts().head(10)
            fig4 = px.bar(x=loc_reqs.values, y=loc_reqs.index, orientation='h', title="311 Requests by Location")
            st.plotly_chart(fig4, width="stretch")
        
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
            st.plotly_chart(fig5, width="stretch")
            
        st.dataframe(filtered_datasets["service"], width="stretch")
    else:
        st.info("No 311 request data available.")

# Map View (New Tab)
with tab6:
    st.subheader("Deep Ellum Interactive Map")
    st.write("This map displays the Deep Ellum area. You can overlay different data types on it.")

    # Load the map image (Corrected path)
    map_image_path = Path(r"C:\Users\juhco\OneDrive\Documents\krowdguide-cleanfinal\data\deepellummap.jpg")
    if map_image_path.exists():
        st.image(map_image_path, caption="Deep Ellum Housing TIF Area", width="stretch")
    else:
        st.warning(f"Map image '{map_image_path}' not found. Please verify the file path.")

    # Example: Displaying a bar chart of arrest locations as a fallback
    if not filtered_datasets["arrests"].empty:
        loc_col = detect_col(filtered_datasets["arrests"], "location", "area", "address")
        if loc_col:
            st.subheader("Top Arrest Locations")
            loc_counts = filtered_datasets["arrests"][loc_col].value_counts().head(10)
            fig = px.bar(x=loc_counts.values, y=loc_counts.index, orientation='h', title="Top 10 Arrest Locations")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No location data found for arrests to display.")
    else:
        st.info("No arrest data available for the map view.")

# Footer
st.divider()
st.caption("KrowdGuide — Transforming Urban Data into Strategic Intelligence | Deep Ellum Focus")