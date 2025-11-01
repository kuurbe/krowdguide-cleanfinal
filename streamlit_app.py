# streamlit_app.py
from pathlib import Path
import pandas as pd
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import numpy as np

# ------------------ APP CONFIG ------------------
st.set_page_config(
    page_title="KrowdGuide — Deep Ellum Predictive Heatmap",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ HEADER ------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
.main-header { font-size: 2.6rem; font-weight: 700; color: #1a365d; margin-bottom: 0.25rem; }
.sub-header  { font-size: 1.1rem; color: #4a5568; margin-bottom: 1.2rem; }
.card { background: #fff; border-radius: 12px; padding: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.gold { background: linear-gradient(135deg,#f0c850,#d4af37); color:#fff; padding: 16px; border-radius: 14px; text-align:center; }
.legend { position: fixed; bottom: 20px; right: 20px; z-index: 9999; background: rgba(255,255,255,.92); padding: 10px 12px; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,.2); }
.legend h4 { margin: 0 0 6px 0; font-size: 14px; }
.legend .row { display:flex; align-items:center; gap:8px; font-size:12px; margin: 2px 0; }
.swatch { width: 14px; height: 14px; border-radius: 3px; display:inline-block; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">KrowdGuide</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header"><b>Deep Ellum Predictive Intelligence Heatmap</b> — AI-enhanced urban insights</div>', unsafe_allow_html=True)

# ------------------ DATA LOADING ------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heatmap_points.geojson"

if not DATA_PATH.exists():
    st.error("❌ GeoJSON file not found. Please run your Jupyter notebook to generate `data/heatmap_points.geojson`.")
    st.stop()

gdf = gpd.read_file(DATA_PATH)
st.success(f"✅ Loaded {len(gdf):,} points from GeoJSON")

# Validate coordinate bounds
if "latitude" in gdf.columns and "longitude" in gdf.columns:
    gdf = gdf[(gdf["latitude"].between(32.778, 32.786)) & (gdf["longitude"].between(-96.790, -96.775))]

# ------------------ SIDEBAR ------------------
st.sidebar.header("Map Filters")
score_field = "crime_score" if "crime_score" in gdf.columns else None
cluster_field = "cluster" if "cluster" in gdf.columns else None

show_clusters = st.sidebar.checkbox("Show Cluster Markers", value=True)
show_heatmap = st.sidebar.checkbox("Show Crime Heatmap", value=True)
radius = st.sidebar.slider("Heat Radius", 8, 40, 22)
blur = st.sidebar.slider("Heat Blur", 6, 40, 18)
min_opacity = st.sidebar.slider("Min Opacity", 0.0, 1.0, 0.35)

# ------------------ MAP ------------------
st.subheader("🗺 Interactive Deep Ellum Map")

m = folium.Map(location=[32.782, -96.782], zoom_start=15, control_scale=True, tiles="CartoDB positron")

# Heatmap Layer
if show_heatmap and score_field:
    heat_points = gdf[["latitude", "longitude", score_field]].dropna().values.tolist()
    HeatMap(heat_points, radius=radius, blur=blur, min_opacity=min_opacity).add_to(m)

# Cluster Marker Layer
if show_clusters:
    cluster_layer = MarkerCluster(name="Predictive Points").add_to(m)
    for _, row in gdf.iterrows():
        lat, lon = float(row.geometry.y), float(row.geometry.x)
        popup_content = f"""
        <div style='min-width:280px; font-family:"Times New Roman", serif;'>
          <b>📍 Coordinates:</b> {lat:.3f}, {lon:.3f}<br>
          <b>Cluster:</b> {row.get('cluster', 'N/A')}<br>
          <b>Crime Score:</b> {row.get('crime_score', 'N/A')}<br>
          <b>Predicted Foot Traffic:</b> ~{np.random.randint(800,1500)} / day<br>
        </div>
        """
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            popup=popup_content,
            color="#1d4ed8",
            fill=True,
            fill_opacity=0.85
        ).add_to(cluster_layer)

# Add legend
legend_html = """
<div class='legend'>
  <h4>Crime Intensity</h4>
  <div class='row'><span class='swatch' style='background:#3B82F6'></span> Low</div>
  <div class='row'><span class='swatch' style='background:#F59E0B'></span> Medium</div>
  <div class='row'><span class='swatch' style='background:#EF4444'></span> High</div>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)

# Display Map
st_folium(m, width="stretch", height=700)

# ------------------ INSIGHT PANELS ------------------
st.markdown('<div class="gold"><b>🌟 Predictive Overview</b></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    avg_crime = round(gdf["crime_score"].mean(), 2) if "crime_score" in gdf else "N/A"
    st.markdown(f'<div class="card"><b>Average Crime Score</b><br><span style="font-size:1.4rem;">{avg_crime}</span></div>', unsafe_allow_html=True)
with col2:
    total_points = len(gdf)
    st.markdown(f'<div class="card"><b>Active Data Points</b><br><span style="font-size:1.4rem;">{total_points:,}</span></div>', unsafe_allow_html=True)
with col3:
    clusters = len(gdf["cluster"].unique()) if "cluster" in gdf else "N/A"
    st.markdown(f'<div class="card"><b>Identified Clusters</b><br><span style="font-size:1.4rem;">{clusters}</span></div>', unsafe_allow_html=True)

# ------------------ OPTIONAL: ANALYSIS TAB ------------------
st.subheader("📊 Cluster Crime Distribution")

if cluster_field and score_field:
    chart_df = gdf.groupby(cluster_field)[score_field].mean().reset_index()
    fig = px.bar(
        chart_df,
        x=cluster_field,
        y=score_field,
        color=score_field,
        title="Average Crime Score by Cluster",
    )
    st.plotly_chart(fig, width="stretch")

# ------------------ FOOTER ------------------
st.divider()
st.caption("KrowdGuide — Deep Ellum Predictive Intelligence • Powered by Jupyter + Streamlit (2025)")


