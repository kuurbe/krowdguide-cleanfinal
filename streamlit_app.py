# streamlit_app.py
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
import streamlit as st
from folium import Map, LayerControl
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px

# ------------------ App Config ------------------
st.set_page_config(
    page_title="KrowdGuide — Deep Ellum Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Styling (Times New Roman + bold headers) ------------------
st.markdown("""
<style>
  html, body, [class*="css"]  { font-family: 'Times New Roman', serif; }
  .main-header { font-size: 2.6rem; font-weight: 700; color: #1a365d; margin: 0 0 .25rem 0; }
  .sub-header  { font-size: 1.15rem; color: #4a5568; margin: 0 0 1.25rem 0; }
  .card { background: #fff; border-radius: 12px; padding: 14px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
  .gold { background: linear-gradient(135deg,#f0c850,#d4af37); color:#fff; padding: 16px; border-radius: 14px; text-align:center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">KrowdGuide</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header"><b>Deep Ellum Intelligence Dashboard</b> — Real Display (uses your local data)</div>', unsafe_allow_html=True)

# ------------------ Paths ------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
if not DATA_DIR.exists():
    st.error("❌ Data folder `data/` not found next to this file.")
    st.stop()

# ------------------ Helpers ------------------
@st.cache_data
def load_all_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    files = sorted(list(data_dir.glob("*.csv")) + list(data_dir.glob("*.csv.gz")))
    ds = {}
    for p in files:
        try:
            df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
            ds[p.name] = df
        except Exception as e:
            ds[p.name] = pd.DataFrame()
    return ds

def detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df.empty: return None
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand = cand.lower()
        for c in cols:
            if cand == c or cand in c or c.startswith(cand):
                return cols[c]
    return None

def deep_ellum_bbox():
    # From your spec
    lat_min, lat_max = 32.778, 32.786
    lon_min, lon_max = -96.790, -96.775
    return lat_min, lat_max, lon_min, lon_max

def within_bbox(lat, lon) -> bool:
    lat_min, lat_max, lon_min, lon_max = deep_ellum_bbox()
    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)

def pick_weight_column(df: pd.DataFrame) -> str | None:
    for k in ["crowd_proxy","weight","count","visits","volume","incidents","value"]:
        c = detect_col(df,[k])
        if c: return c
    return None

def extract_points(df: pd.DataFrame, lat_col: str, lon_col: str, weight_col: str | None):
    # Clean numeric
    d = df[[lat_col, lon_col] + ([weight_col] if weight_col else [])].copy()
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
    if weight_col:
        d[weight_col] = pd.to_numeric(d[weight_col], errors="coerce").fillna(1.0)
    d = d.dropna(subset=[lat_col, lon_col])
    # BBox filter to Deep Ellum
    lat_min, lat_max, lon_min, lon_max = deep_ellum_bbox()
    d = d[(d[lat_col].between(lat_min, lat_max)) & (d[lon_col].between(lon_min, lon_max))]
    # Build (lat, lon, weight)
    if weight_col:
        pts = d.apply(lambda r: [float(r[lat_col]), float(r[lon_col]), float(max(r[weight_col], 0.1))], axis=1).tolist()
    else:
        pts = d.apply(lambda r: [float(r[lat_col]), float(r[lon_col]), 1.0], axis=1).tolist()
    return pts

# ------------------ Load Data ------------------
datasets = load_all_csvs(DATA_DIR)
if not datasets:
    st.warning("⚠️ No CSVs found in `data/`.")
    st.stop()

st.markdown('<div class="gold"><b>🌟 Real Display:</b> Using your local CSVs (no external calls)</div>', unsafe_allow_html=True)

# Sidebar – choose dataset for heatmap
st.sidebar.header("Map Settings")
ds_names = [name for name, df in datasets.items() if not df.empty]
if not ds_names:
    st.info("No non-empty CSVs found.")
    st.stop()

chosen_name = st.sidebar.selectbox("Choose dataset for Heatmap", ds_names, index=0)
chosen_df = datasets[chosen_name]

# Try to find coordinate columns
lat_col = detect_col(chosen_df, ["lat", "latitude", "y", "y_coord", "ycoordinate"])
lon_col = detect_col(chosen_df, ["lon", "lng", "longitude", "x", "x_coord", "xcoordinate"])
weight_col = pick_weight_column(chosen_df)

# Date filters (optional)
date_col = detect_col(chosen_df, ["date", "datetime", "timestamp"])
if date_col:
    with st.sidebar.expander("Date Filter", expanded=False):
        try:
            sdf = chosen_df.copy()
            sdf[date_col] = pd.to_datetime(sdf[date_col], errors="coerce")
            sdf = sdf.dropna(subset=[date_col])
            min_d, max_d = sdf[date_col].min(), sdf[date_col].max()
            start, end = st.date_input(
                "Range", value=[min_d.date(), max_d.date()],
                min_value=min_d.date(), max_value=max_d.date()
            )
            mask = (sdf[date_col] >= pd.Timestamp(start)) & (sdf[date_col] <= pd.Timestamp(end))
            chosen_df = sdf.loc[mask].copy()
        except Exception:
            pass

# ------------------ Tabs ------------------
tab_map, tab_overview, tab_table = st.tabs(["🗺 Heatmap", "📊 Overview", "🔎 Data Table"])

# ------------------ Heatmap Tab ------------------
with tab_map:
    st.subheader("Real-Time Crowd/Activity Heatmap (from selected dataset)")
    if lat_col and lon_col:
        # Controls
        radius = st.sidebar.slider("Heat radius", 5, 40, 20)
        blur   = st.sidebar.slider("Heat blur", 5, 40, 15)
        min_op = st.sidebar.slider("Min opacity", 0.0, 1.0, 0.3)

        points = extract_points(chosen_df, lat_col, lon_col, weight_col)
        st.caption(f"Dataset: **{chosen_name}** | Points used: **{len(points):,}** | Lat: **{lat_col}**, Lon: **{lon_col}**" + (f" | Weight: **{weight_col}**" if weight_col else ""))

        # Build map
        m = Map(location=[32.783, -96.783], zoom_start=15, control_scale=True, prefer_canvas=True)
        if points:
            HeatMap(points, radius=radius, blur=blur, min_opacity=min_op, max_zoom=18).add_to(m)
        LayerControl(position="topleft").add_to(m)
        st_folium(m, height=650, width=1200)
    else:
        st.warning("Couldn’t find latitude/longitude columns in this dataset. Try another file or rename columns to include ‘lat’ & ‘lon’.")

# ------------------ Overview Tab ------------------
with tab_overview:
    st.subheader("Quick Overview")
    # Simple summaries that won’t crash
    cards = st.columns(3)
    with cards[0]:
        st.markdown('<div class="card"><b>Total records</b><br><span style="font-size:1.5rem;">{:,}</span></div>'.format(len(chosen_df)), unsafe_allow_html=True)
    with cards[1]:
        if weight_col and not chosen_df.empty:
            try:
                total_w = pd.to_numeric(chosen_df[weight_col], errors="coerce").fillna(0).clip(lower=0).sum()
                st.markdown('<div class="card"><b>Total weight</b><br><span style="font-size:1.5rem;">{:,}</span></div>'.format(int(total_w)), unsafe_allow_html=True)
            except Exception:
                st.markdown('<div class="card"><b>Total weight</b><br><span style="font-size:1.5rem;">N/A</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><b>Total weight</b><br><span style="font-size:1.5rem;">N/A</span></div>', unsafe_allow_html=True)
    with cards[2]:
        if date_col and not chosen_df.empty:
            try:
                dmin = pd.to_datetime(chosen_df[date_col], errors="coerce").min()
                dmax = pd.to_datetime(chosen_df[date_col], errors="coerce").max()
                st.markdown(f'<div class="card"><b>Date span</b><br><span style="font-size:1.5rem;">{dmin.date()} → {dmax.date()}</span></div>', unsafe_allow_html=True)
            except Exception:
                st.markdown('<div class="card"><b>Date span</b><br><span style="font-size:1.5rem;">N/A</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><b>Date span</b><br><span style="font-size:1.5rem;">N/A</span></div>', unsafe_allow_html=True)

    # Optional simple chart (by day) if date & weight present
    if date_col and not chosen_df.empty:
        try:
            tmp = chosen_df.copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col])
            if weight_col:
                tmp["w"] = pd.to_numeric(tmp[weight_col], errors="coerce").fillna(0).clip(lower=0)
                agg = tmp.groupby(tmp[date_col].dt.date)["w"].sum().reset_index(name="value")
            else:
                agg = tmp.groupby(tmp[date_col].dt.date).size().reset_index(name="value")
            fig = px.line(agg, x=agg.columns[0], y="value", title="Daily Activity (count/weight)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

# ------------------ Data Table Tab ------------------
with tab_table:
    st.subheader(f"Dataset Preview — {chosen_name}")
    st.dataframe(chosen_df.head(1000), use_container_width=True)

st.divider()
st.caption("KrowdGuide — Transforming Urban Data into Strategic Intelligence | Deep Ellum Focus")

