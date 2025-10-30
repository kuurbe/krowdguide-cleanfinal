# streamlit_app.py
import os
from pathlib import Path
from datetime import datetime, timedelta
import io

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import humanize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ------------------ App Config ------------------
st.set_page_config(
    page_title="KrowdGuide Smart City OS",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🏙️ KrowdGuide — Smart City OS (Prototype)")

# Render dynamic port (safe to ignore locally)
_ = int(os.environ.get("PORT", 8501))

BASE_DIR = Path(__file__).resolve().parent
DATA_DIRS = [
    BASE_DIR / "data",  # repo-relative
    Path(r"C:\Users\juhco\OneDrive\Documents\krowdguide-cleanfinal\data"),  # Windows dev fallback
]

def first_existing_dir(paths):
    for p in paths:
        if p and p.exists():
            return p
    return None

DATA_DIR = first_existing_dir(DATA_DIRS)
if DATA_DIR is None:
    st.error("❌ Could not find a `data/` folder. Create one next to this file or set a valid path.")
    st.stop()

# ------------------ Discovery ------------------
def list_csvs(data_dir: Path):
    # Support .csv and compressed .csv.gz
    return sorted(list(data_dir.glob("*.csv")) + list(data_dir.glob("*.csv.gz")))

CSV_PATHS = list_csvs(DATA_DIR)
if not CSV_PATHS:
    st.warning(f"⚠️ No CSV files found in: {DATA_DIR}")
    uploaded = st.file_uploader("Upload a CSV to begin", type=["csv"])
    if uploaded:
        try:
            tmp = pd.read_csv(uploaded, low_memory=False, on_bad_lines="skip")
            st.success("✅ Uploaded. Preview below:")
            st.dataframe(tmp.head(100), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
    st.stop()

# ------------------ Helpers ------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Robust CSV load with fallbacks; cached by file path + mtime + size."""
    try:
        if str(path).endswith(".gz"):
            return pd.read_csv(path, compression="infer", low_memory=False, on_bad_lines="skip")
        return pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    except Exception:
        # Fallback to python engine
        return pd.read_csv(path, low_memory=False, on_bad_lines="skip", engine="python")

@st.cache_data
def file_fingerprint(path: Path):
    s = path.stat()
    return (str(path), s.st_size, s.st_mtime)

def info_for(path: Path):
    s = path.stat()
    return {
        "file": path.name,
        "size": s.st_size,
        "size_readable": humanize.naturalsize(s.st_size, binary=False),
        "modified": datetime.fromtimestamp(s.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "path": str(path),
    }

def detect_col(df: pd.DataFrame, *keys, prefer_startswith=False, contains=True):
    """Find first column that matches any of the keys; flexible matching."""
    cols = [str(c) for c in df.columns]
    lower = {c.lower(): c for c in cols}
    for key in keys:
        k = key.lower()
        if prefer_startswith:
            for c in cols:
                if c.lower().startswith(k):
                    return c
        if contains:
            for c in cols:
                if k in c.lower():
                    return c
        if k in lower:
            return lower[k]
    return None

def figure():
    fig, ax = plt.subplots()
    return fig, ax

def preview_and_download(df: pd.DataFrame, filename: str, key_suffix: str = ""):
    st.caption("Preview limited for speed. Download always includes **all** rows.")
    n = st.slider("Rows to preview", 10, 2000, 100, step=10, key=f"preview_{filename}_{key_suffix}")
    st.dataframe(df.head(n), use_container_width=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download FULL dataset (CSV)",
        data=csv_bytes,
        file_name=filename if filename.endswith(".csv") else f"{filename}.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"dl_{filename}_{key_suffix}"
    )

def pick_by_name(substr: str):
    for p in CSV_PATHS:
        if substr.lower().replace(" ", "") in p.name.lower().replace(" ", ""):
            return p
    return None

def predict_weather(df, date_col, temp_col, days=30):
    """Generate weather predictions using polynomial regression"""
    if date_col and temp_col:
        df_clean = df[[date_col, temp_col]].dropna()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        df_clean = df_clean.sort_values(date_col)
        
        # Convert dates to numeric for regression
        df_clean['days'] = (df_clean[date_col] - df_clean[date_col].min()).dt.days
        
        # Prepare data
        X = df_clean[['days']].values
        y = df_clean[temp_col].values
        
        # Train polynomial model
        model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        model.fit(X, y)
        
        # Generate future dates
        last_date = df_clean[date_col].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
        future_days = [(d - df_clean[date_col].min()).days for d in future_dates]
        future_X = np.array(future_days).reshape(-1, 1)
        
        # Predict
        predictions = model.predict(future_X)
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            date_col: future_dates,
            temp_col: predictions,
            'type': 'predicted'
        })
        
        # Add actual data
        actual_df = df_clean[[date_col, temp_col]].copy()
        actual_df['type'] = 'actual'
        
        return pd.concat([actual_df, pred_df], ignore_index=True)
    return df

# ------------------ Named datasets (best-effort) ------------------
PATH_VISITS   = pick_by_name("DeepEllumVisits") or pick_by_name("WeeklyVisits")
PATH_ARRESTS  = pick_by_name("arrests")
PATH_SERVICE  = pick_by_name("service requests") or pick_by_name("311")
PATH_TRAFFIC  = pick_by_name("traffic") or pick_by_name("daily_clean")
PATH_BIKE_PED = pick_by_name("bike_pedestrian") or pick_by_name("bike") or pick_by_name("pedestrian")
PATH_WEATHER  = pick_by_name("DeepWeather") or pick_by_name("weather")  # Updated filename
PATH_TXDOT    = pick_by_name("TxDOT")
PATH_TREES    = pick_by_name("Trees")

# ------------------ Sidebar: Navigation & Catalog ------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Mobility", "Public Safety", "Infrastructure", "Sustainability", "Data Explorer"],
)

st.sidebar.markdown(f"**Data folder:** `{DATA_DIR}`")
catalog = pd.DataFrame([info_for(p) for p in CSV_PATHS])
with st.sidebar.expander("Datasets (quick view)"):
    st.dataframe(catalog[["file", "size_readable", "modified"]], use_container_width=True, hide_index=True)

# ------------------ DASHBOARD ------------------
if page == "Dashboard":
    st.subheader("📊 City Intelligence Overview")
    
    # Load all data at once
    data_sources = {
        'visits': (PATH_VISITS, "Weekly Visits"),
        'arrests': (PATH_ARRESTS, "Arrests"),
        'service': (PATH_SERVICE, "Service Requests"),
        'traffic': (PATH_TRAFFIC, "Traffic Records"),
        'weather': (PATH_WEATHER, "Weather Data"),
        'bike_ped': (PATH_BIKE_PED, "Bike/Ped Traffic"),
        'txdot': (PATH_TXDOT, "TxDOT Data")
    }
    
    loaded_data = {}
    for key, (path, name) in data_sources.items():
        if path:
            loaded_data[key] = load_csv(path)
        else:
            loaded_data[key] = pd.DataFrame()
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        vcol = detect_col(loaded_data['visits'], "visits", "count")
        last_visits = int(loaded_data['visits'][vcol].iloc[-1]) if (not loaded_data['visits'].empty and vcol) else 0
        st.metric("Weekly Visits (last row)", f"{last_visits:,}")
    with k2:
        st.metric("Arrests (rows)", f"{len(loaded_data['arrests']):,}")
    with k3:
        st.metric("Service Requests (rows)", f"{len(loaded_data['service']):,}")
    with k4:
        st.metric("Traffic Records (rows)", f"{len(loaded_data['traffic']):,}")

    # Combined visualization section
    st.markdown("#### Traffic & Weather Correlation")
    if not loaded_data['traffic'].empty and not loaded_data['weather'].empty:
        date_col_traffic = detect_col(loaded_data['traffic'], "date", "day", "datetime", prefer_startswith=True)
        vol_col = detect_col(loaded_data['traffic'], "volume", "count", "avg")
        date_col_weather = detect_col(loaded_data['weather'], "date", "day", "datetime", prefer_startswith=True)
        temp_col = detect_col(loaded_data['weather'], "temperature", "temp", "temp_f", "temp_c")
        
        if date_col_traffic and vol_col and date_col_weather and temp_col:
            # Merge data on date
            traffic_df = loaded_data['traffic'][[date_col_traffic, vol_col]].rename(columns={date_col_traffic: "date", vol_col: "traffic_volume"})
            weather_df = loaded_data['weather'][[date_col_weather, temp_col]].rename(columns={date_col_weather: "date", temp_col: "temperature"})
            merged_df = pd.merge(traffic_df, weather_df, on="date", how="inner")
            
            if not merged_df.empty:
                fig = px.scatter(merged_df, x="traffic_volume", y="temperature", 
                                title="Traffic Volume vs Temperature",
                                trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No overlapping dates found between traffic and weather data")
        else:
            st.info("Required columns not found for traffic/weather correlation")
    
    # Foot Traffic Trend
    if not loaded_data['visits'].empty:
        st.markdown("#### Foot Traffic Trend")
        wk = detect_col(loaded_data['visits'], "week_start_iso", "week")
        vcol = detect_col(loaded_data['visits'], "visits")
        venue = detect_col(loaded_data['visits'], "venue")
        if wk and vcol:
            fig = px.line(loaded_data['visits'], x=wk, y=vcol, color=venue,
                         title="Foot Traffic Over Time",
                         markers=True)
            fig.update_layout(xaxis_title="Week", yaxis_title="Visits")
            st.plotly_chart(fig, use_container_width=True)

    # Top Service/Request Categories
    if not loaded_data['service'].empty:
        st.markdown("#### Top Service/Request Categories")
        req = detect_col(loaded_data['service'], "request", "service", "topic")
        if req:
            vc = loaded_data['service'][req].astype("string").str.strip().fillna("⟂ NA").value_counts().head(10).reset_index()
            vc.columns = [req, "count"]
            fig = px.bar(vc, x="count", y=req, orientation='h',
                        title="Top Service Requests",
                        labels={req: "Request Type", "count": "Count"})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    # Weather Snapshot with Predictions
    if not loaded_data['weather'].empty:
        st.markdown("#### Weather Forecast")
        dcol = detect_col(loaded_data['weather'], "date", "datetime", "timestamp", prefer_startswith=True)
        tcol = detect_col(loaded_data['weather'], "temperature", "temp", "temp_f", "temp_c")
        pcol = detect_col(loaded_data['weather'], "precip", "rain", "precipitation")
        hcol = detect_col(loaded_data['weather'], "humidity")
        
        if dcol and tcol:
            # Predict weather
            pred_df = predict_weather(loaded_data['weather'], dcol, tcol)
            
            fig = px.line(pred_df, x=dcol, y=tcol, color='type',
                         title="Temperature Forecast",
                         labels={dcol: "Date", tcol: "Temperature (°F)"})
            fig.update_layout(xaxis_title="Date", yaxis_title="Temperature (°F)")
            st.plotly_chart(fig, use_container_width=True)
        
        cols = st.columns(3)
        with cols[0]:
            if tcol: st.metric("Last Temperature", f"{loaded_data['weather'][tcol].dropna().iloc[-1]:.1f}°F")
        with cols[1]:
            if pcol: st.metric("Last Precip", f"{loaded_data['weather'][pcol].dropna().iloc[-1]:.2f}in")
        with cols[2]:
            if hcol: st.metric("Last Humidity", f"{loaded_data['weather'][hcol].dropna().iloc[-1]:.0f}%")

# ------------------ MOBILITY ------------------
elif page == "Mobility":
    st.subheader("🚶 Mobility: Traffic, Bike/Ped, Footfall")
    
    # Load all mobility data
    mobility_data = {}
    for key, path in [('visits', PATH_VISITS), ('traffic', PATH_TRAFFIC), ('bike_ped', PATH_BIKE_PED), ('txdot', PATH_TXDOT)]:
        if path:
            mobility_data[key] = load_csv(path)
        else:
            mobility_data[key] = pd.DataFrame()
    
    # Summary metrics
    st.markdown("#### Mobility Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Foot Traffic Records", f"{len(mobility_data['visits']):,}")
    with m2:
        st.metric("Traffic Records", f"{len(mobility_data['traffic']):,}")
    with m3:
        st.metric("Bike/Ped Records", f"{len(mobility_data['bike_ped']):,}")
    with m4:
        st.metric("TxDOT Records", f"{len(mobility_data['txdot']):,}")
    
    # Combined mobility chart
    st.markdown("#### Combined Mobility Trends")
    
    # Foot Traffic
    if not mobility_data['visits'].empty:
        wk = detect_col(mobility_data['visits'], "week_start_iso", "week")
        vcol = detect_col(mobility_data['visits'], "visits")
        venue = detect_col(mobility_data['visits'], "venue")
        if wk and vcol:
            fig = px.line(mobility_data['visits'], x=wk, y=vcol, color=venue,
                         title="Foot Traffic Over Time",
                         markers=True)
            fig.update_layout(xaxis_title="Week", yaxis_title="Visits")
            st.plotly_chart(fig, use_container_width=True)
    
    # Traffic Volume
    if not mobility_data['traffic'].empty:
        date_col = detect_col(mobility_data['traffic'], "date", "day", prefer_startswith=True)
        vol_col = detect_col(mobility_data['traffic'], "volume", "count", "avg")
        if date_col and vol_col:
            fig = px.line(mobility_data['traffic'], x=date_col, y=vol_col,
                         title="Traffic Volume Over Time",
                         markers=True)
            fig.update_layout(xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig, use_container_width=True)
    
    # Bike/Ped Traffic
    if not mobility_data['bike_ped'].empty:
        x = detect_col(mobility_data['bike_ped'], "date", "day", prefer_startswith=True)
        y = detect_col(mobility_data['bike_ped'], "count", "avg", "volume")
        loc = detect_col(mobility_data['bike_ped'], "location", "area", "zone")
        if x and y:
            if loc:
                fig = px.line(mobility_data['bike_ped'], x=x, y=y, color=loc,
                             title="Bike/Ped Traffic by Location",
                             markers=True)
            else:
                fig = px.line(mobility_data['bike_ped'], x=x, y=y,
                             title="Bike/Ped Counts Over Time",
                             markers=True)
            fig.update_layout(xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    
    # TxDOT Data
    if not mobility_data['txdot'].empty:
        date_col = detect_col(mobility_data['txdot'], "date", "week", "datetime", prefer_startswith=True)
        metric = detect_col(mobility_data['txdot'], "count", "closures", "incidents", "events")
        loc = detect_col(mobility_data['txdot'], "location", "area", "zone")
        if date_col and metric:
            if loc:
                fig = px.line(mobility_data['txdot'], x=date_col, y=metric, color=loc,
                             title="TxDOT Incidents by Location",
                             markers=True)
            else:
                fig = px.line(mobility_data['txdot'], x=date_col, y=metric,
                             title="TxDOT Incidents Over Time",
                             markers=True)
            fig.update_layout(xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

# ------------------ PUBLIC SAFETY ------------------
elif page == "Public Safety":
    st.subheader("👮 Public Safety: Arrests & Requests")
    
    # Load all safety data
    safety_data = {}
    for key, path in [('arrests', PATH_ARRESTS), ('service', PATH_SERVICE)]:
        if path:
            safety_data[key] = load_csv(path)
        else:
            safety_data[key] = pd.DataFrame()
    
    # Summary metrics
    st.markdown("#### Public Safety Metrics")
    s1, s2 = st.columns(2)
    with s1:
        st.metric("Arrest Records", f"{len(safety_data['arrests']):,}")
    with s2:
        st.metric("Service Requests", f"{len(safety_data['service']):,}")
    
    # Combined safety chart
    st.markdown("#### Public Safety Trends")
    
    # Arrests
    if not safety_data['arrests'].empty:
        cat = detect_col(safety_data['arrests'], "offense", "charge", "category")
        date_col = detect_col(safety_data['arrests'], "date", "arrest", "datetime", prefer_startswith=True)
        loc = detect_col(safety_data['arrests'], "location", "area", "zone")
        
        if cat:
            st.markdown("**Top Offenses**")
            vc = safety_data['arrests'][cat].astype("string").str.strip().fillna("⟂ NA").value_counts().head(15).reset_index()
            vc.columns = [cat, "count"]
            fig = px.bar(vc, x="count", y=cat, orientation='h',
                        title="Top Offenses",
                        labels={cat: "Offense Type", "count": "Count"})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        if date_col:
            st.markdown("**Arrests Over Time**")
            ts = safety_data['arrests'].groupby(date_col).size().reset_index(name="count")
            fig = px.line(ts, x=date_col, y="count",
                         title="Arrests Over Time",
                         markers=True)
            fig.update_layout(xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        if loc:
            st.markdown("**Arrests by Location**")
            loc_counts = safety_data['arrests'][loc].value_counts().reset_index()
            loc_counts.columns = [loc, "count"]
            fig = px.bar(loc_counts, x="count", y=loc, orientation='h',
                        title="Arrests by Location",
                        labels={loc: "Location", "count": "Count"})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Service Requests
    if not safety_data['service'].empty:
        req = detect_col(safety_data['service'], "request", "service", "topic")
        date_col = detect_col(safety_data['service'], "date", "week", "datetime", prefer_startswith=True)
        loc = detect_col(safety_data['service'], "location", "area", "zone")
        
        if req:
            st.markdown("**Top Request/Service Categories**")
            vc = safety_data['service'][req].astype("string").str.strip().fillna("⟂ NA").value_counts().head(15).reset_index()
            vc.columns = [req, "count"]
            fig = px.bar(vc, x="count", y=req, orientation='h',
                        title="Top Service Requests",
                        labels={req: "Request Type", "count": "Count"})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        if date_col and req:
            st.markdown("**Requests Trend**")
            trend = (safety_data['service']
                    .assign(__cat=safety_data['service'][req].astype("string").str.strip().fillna("⟂ NA"))
                    .groupby([date_col, "__cat"]).size().reset_index(name="count"))
            fig = px.line(trend, x=date_col, y="count", color="__cat",
                         title="Service Requests Over Time",
                         markers=True)
            fig.update_layout(xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        if loc:
            st.markdown("**Service Requests by Location**")
            loc_counts = safety_data['service'][loc].value_counts().reset_index()
            loc_counts.columns = [loc, "count"]
            fig = px.bar(loc_counts, x="count", y=loc, orientation='h',
                        title="Service Requests by Location",
                        labels={loc: "Location", "count": "Count"})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

# ------------------ INFRASTRUCTURE ------------------
elif page == "Infrastructure":
    st.subheader("🛠️ Infrastructure: 311 & TxDOT")
    
    # Load all infrastructure data
    infra_data = {}
    for key, path in [('service', PATH_SERVICE), ('txdot', PATH_TXDOT)]:
        if path:
            infra_data[key] = load_csv(path)
        else:
            infra_data[key] = pd.DataFrame()
    
    # Summary metrics
    st.markdown("#### Infrastructure Metrics")
    i1, i2 = st.columns(2)
    with i1:
        st.metric("311 Service Requests", f"{len(infra_data['service']):,}")
    with i2:
        st.metric("TxDOT Records", f"{len(infra_data['txdot']):,}")
    
    # Combined infrastructure chart
    st.markdown("#### Infrastructure Trends")
    
    # 311 Data
    if not infra_data['service'].empty:
        topic = detect_col(infra_data['service'], "request", "service", "topic")
        status = detect_col(infra_data['service'], "status", "state")
        loc = detect_col(infra_data['service'], "location", "area", "zone")
        
        if topic:
            st.markdown("**311 Topics**")
            vc = infra_data['service'][topic].astype("string").fillna("⟂ NA").value_counts().head(15).reset_index()
            vc.columns = [topic, "count"]
            fig = px.bar(vc, x="count", y=topic, orientation='h',
                        title="Top 311 Topics",
                        labels={topic: "Topic", "count": "Count"})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        if status:
            st.markdown("**Status Distribution**")
            vc = infra_data['service'][status].astype("string").fillna("⟂ NA").value_counts().reset_index()
            vc.columns = [status, "count"]
            fig = px.pie(vc, values="count", names=status,
                        title="311 Status Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        if loc:
            st.markdown("**311 Requests by Location**")
            loc_counts = infra_data['service'][loc].value_counts().reset_index()
            loc_counts.columns = [loc, "count"]
            fig = px.bar(loc_counts, x="count", y=loc, orientation='h',
                        title="311 Requests by Location",
                        labels={loc: "Location", "count": "Count"})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # TxDOT Data
    if not infra_data['txdot'].empty:
        date_col = detect_col(infra_data['txdot'], "date", "week", "datetime", prefer_startswith=True)
        metric = detect_col(infra_data['txdot'], "count", "closures", "incidents", "events")
        loc = detect_col(infra_data['txdot'], "location", "area", "zone")
        
        if date_col and metric:
            fig = px.line(infra_data['txdot'], x=date_col, y=metric,
                         title="TxDOT Incidents Over Time",
                         markers=True)
            fig.update_layout(xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        if loc:
            st.markdown("**TxDOT Incidents by Location**")
            loc_counts = infra_data['txdot'][loc].value_counts().reset_index()
            loc_counts.columns = [loc, "count"]
            fig = px.bar(loc_counts, x="count", y=loc, orientation='h',
                        title="TxDOT Incidents by Location",
                        labels={loc: "Location", "count": "Count"})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

# ------------------ SUSTAINABILITY (incl. WEATHER) ------------------
elif page == "Sustainability":
    st.subheader("🌳 Sustainability & Weather Insights")
    
    # Load all sustainability data
    sustainability_data = {}
    for key, path in [('trees', PATH_TREES), ('weather', PATH_WEATHER)]:
        if path:
            sustainability_data[key] = load_csv(path)
        else:
            sustainability_data[key] = pd.DataFrame()
    
    # Summary metrics
    st.markdown("#### Sustainability Metrics")
    s1, s2 = st.columns(2)
    with s1:
        st.metric("Tree Records", f"{len(sustainability_data['trees']):,}")
    with s2:
        st.metric("Weather Records", f"{len(sustainability_data['weather']):,}")
    
    # Combined sustainability chart
    st.markdown("#### Sustainability Trends")
    
    # Trees & Green Data
    if not sustainability_data['trees'].empty:
        metric = detect_col(sustainability_data['trees'], "tree", "species", "green", "park")
        count_col = detect_col(sustainability_data['trees'], "count", "number", "qty")
        loc = detect_col(sustainability_data['trees'], "location", "area", "zone")
        
        if metric:
            vc = sustainability_data['trees'][metric].astype("string").fillna("⟂ NA").value_counts().head(20).reset_index()
            vc.columns = [metric, "count"]
            fig = px.bar(vc, x="count", y=metric, orientation='h',
                        title="Top Tree/Green Categories",
                        labels={metric: "Category", "count": "Count"})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Weather Data with Predictions
    if not sustainability_data['weather'].empty:
        dcol = detect_col(sustainability_data['weather'], "datetime", "timestamp", "date", "time", prefer_startswith=True)
        tcol = detect_col(sustainability_data['weather'], "temperature", "temp", "temp_f", "temp_c")
        pcol = detect_col(sustainability_data['weather'], "precip", "rain", "precipitation")
        hcol = detect_col(sustainability_data['weather'], "humidity", "rh")
        
        # Try to parse datetimes if needed
        if dcol and not np.issubdtype(sustainability_data['weather'][dcol].dtype, np.datetime64):
            with st.spinner("Parsing datetimes..."):
                try:
                    sustainability_data['weather'][dcol] = pd.to_datetime(sustainability_data['weather'][dcol], errors="coerce")
                except Exception:
                    pass
        
        # Generate predictions
        if dcol and tcol:
            pred_df = predict_weather(sustainability_data['weather'], dcol, tcol)
            fig = px.line(pred_df, x=dcol, y=tcol, color='type',
                         title="Temperature Forecast",
                         labels={dcol: "Date", tcol: "Temperature (°F)"})
            fig.update_layout(xaxis_title="Date", yaxis_title="Temperature (°F)")
            st.plotly_chart(fig, use_container_width=True)
        
        cols = st.columns(3)
        with cols[0]:
            if tcol and not sustainability_data['weather'][tcol].dropna().empty:
                st.metric("Last Temperature", f"{sustainability_data['weather'][tcol].dropna().iloc[-1]:.1f}°F")
        with cols[1]:
            if pcol and not sustainability_data['weather'][pcol].dropna().empty:
                st.metric("Last Precip", f"{sustainability_data['weather'][pcol].dropna().iloc[-1]:.2f}in")
        with cols[2]:
            if hcol and not sustainability_data['weather'][hcol].dropna().empty:
                st.metric("Last Humidity", f"{sustainability_data['weather'][hcol].dropna().iloc[-1]:.0f}%")

# ------------------ DATA EXPLORER ------------------
elif page == "Data Explorer":
    st.subheader("📁 Data Explorer")
    csv_files = [p.name for p in CSV_PATHS]
    pick = st.selectbox("Select a dataset", csv_files)
    path = DATA_DIR / pick
    fp = file_fingerprint(path)  # cache key
    df = load_csv(path)

    top, side = st.columns([2, 1])
    with side:
        st.write("**Shape**", f"{len(df):,} rows × {df.shape[1]:,} columns")
        mem = df.memory_usage(deep=True).sum()
        st.write("**Memory**", humanize.naturalsize(mem, binary=False))
        st.write("**Columns**", len(df.columns))
    with top:
        preview_and_download(df, pick, key_suffix="explorer")

    st.markdown("### 🧬 Schema & Missingness")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
        st.dataframe(dtypes_df, use_container_width=True)
    with c2:
        miss = df.isna().sum()
        if miss.sum() == 0:
            st.write("✅ No missing values.")
        else:
            mt = (
                miss.reset_index()
                .rename(columns={"index": "column", 0: "missing"})
                .assign(total=len(df), pct=lambda x: (x["missing"] / x["total"] * 100).round(2))
                .sort_values("pct", ascending=False)
            )
            st.dataframe(mt, use_container_width=True)

    st.markdown("### 📊 Quick Summaries")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("**Numeric summary**")
        if num_cols:
            st.dataframe(df[num_cols].describe().T, use_container_width=True)
        else:
            st.write("No numeric columns detected.")
    with s2:
        st.markdown("**Top categories (first 6 columns)**")
        if cat_cols:
            tabs = st.tabs(cat_cols[:6])
            for tab, col in zip(tabs, cat_cols[:6]):
                with tab:
                    vc = (
                        df[col].astype("string").fillna("⟂ NA")
                        .value_counts(dropna=False)
                        .head(10)
                        .reset_index()
                    )
                    vc.columns = [col, "count"]
                    st.dataframe(vc, use_container_width=True)
        else:
            st.write("No categorical columns detected.")

# --------------- END ---------------
st.caption(f"📂 Data directory: `{DATA_DIR}` · {len(CSV_PATHS)} file(s) detected · Preview is limited for speed; downloads are full.")
EOF

# Initialize git repo and push
git init
git add .
git commit -m "Enhanced Smart City OS with automated reporting and weather predictions"
git remote add origin https://github.com//kuurbe
git branch -M main
git push -u origin main