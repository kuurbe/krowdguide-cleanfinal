# streamlit_app.py
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
from folium import Map, LayerControl, CircleMarker, FeatureGroup
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px

# ------------------ App Config ------------------
st.set_page_config(
    page_title="KrowdGuide — Deep Ellum Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Styling ------------------
st.markdown("""
<style>
  html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
  .main-header { font-size: 2.4rem; font-weight: 700; color: #1a365d; margin: 0 0 .25rem 0; }
  .sub-header  { font-size: 1.08rem; color: #4a5568; margin: 0 0 1rem 0; }
  .card { background: #fff; border-radius: 12px; padding: 14px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
  .gold { background: linear-gradient(135deg,#f0c850,#d4af37); color:#fff; padding: 16px; border-radius: 14px; text-align:center; }
  .legend { position: fixed; bottom: 22px; right: 22px; z-index: 9999; background: rgba(255,255,255,.92); padding: 10px 12px; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,.2); }
  .legend h4 { margin: 0 0 6px 0; font-size: 14px; }
  .legend .row { display:flex; align-items:center; gap:8px; font-size:12px; margin: 2px 0; }
  .swatch { width: 14px; height: 14px; border-radius: 3px; display:inline-block; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">KrowdGuide</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header"><b>Deep Ellum All-in-One Heatmap</b> — clickable dots with predictive popup</div>', unsafe_allow_html=True)

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
        except Exception:
            ds[p.name] = pd.DataFrame()
    return ds

def detect_col(df: pd.DataFrame, candidates) -> str | None:
    if df.empty: return None
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = cand.lower()
        for k, orig in lower.items():
            if c == k or c in k or k.startswith(c):
                return orig
    return None

def deep_ellum_bbox():
    # Project’s agreed bounds
    lat_min, lat_max = 32.778, 32.786
    lon_min, lon_max = -96.790, -96.775
    return lat_min, lat_max, lon_min, lon_max

def filter_bbox(df, lat_col, lon_col):
    lat_min, lat_max, lon_min, lon_max = deep_ellum_bbox()
    d = df.copy()
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
    d = d.dropna(subset=[lat_col, lon_col])
    return d[(d[lat_col].between(lat_min, lat_max)) & (d[lon_col].between(lon_min, lon_max))]

def round_coord(df, lat_col, lon_col, prec=3):
    d = df.copy()
    d["_lat_r"] = d[lat_col].round(prec)
    d["_lon_r"] = d[lon_col].round(prec)
    d["_grid_id"] = d["_lat_r"].astype(str) + "," + d["_lon_r"].astype(str)
    return d

def inferred_zip(df):
    # Use ZIP column if present; otherwise default to 75226 for Deep Ellum
    zip_col = detect_col(df, ["zip", "zipcode", "postal"])
    if zip_col: return df[zip_col].astype(str)
    return pd.Series(["75226"] * len(df), index=df.index, name="zip")

def predict_avg_foot_traffic(visits_df) -> float:
    """Return a simple predicted average visits (robust fallback)."""
    if visits_df is None or visits_df.empty:
        # Fallback “realistic” baseline for demo if no visits
        return float(1200)
    date_col = detect_col(visits_df, ["date","week","datetime","timestamp"])
    vcol     = detect_col(visits_df, ["visits","count","foot","volume"])
    if not date_col or not vcol: 
        return float(visits_df.shape[0]) if visits_df.shape[0] > 0 else 1200.0
    tmp = visits_df[[date_col, vcol]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[vcol] = pd.to_numeric(tmp[vcol], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return 1200.0
    # 7-day rolling mean as a simple predictor
    s = tmp.set_index(date_col)[vcol].sort_index()
    s = s.rolling(7, min_periods=1).mean()
    return float(np.nanmean(s.tail(14))) if not s.empty else float(np.nanmean(tmp[vcol]))

def build_crime_score(crime_df, arrests_df, service_df):
    """Return a per-grid crime score and arrests tally using rounded lat/lon grouping."""
    parts = []

    def piece(df, weight):
        if df is None or df.empty: return None
        lat = detect_col(df, ["lat","latitude","y"])
        lon = detect_col(df, ["lon","lng","longitude","x"])
        if not lat or not lon: return None
        x = filter_bbox(df, lat, lon)
        if x.empty: return None
        x = round_coord(x, lat, lon, prec=3)
        g = x.groupby("_grid_id").size().rename("count").reset_index()
        g["weighted"] = g["count"] * weight
        return g

    # You can tune these weights
    if crime_df is not None:
        parts.append(piece(crime_df, 1.0))
    if arrests_df is not None:
        parts.append(piece(arrests_df, 2.0))
    if service_df is not None:
        parts.append(piece(service_df, 0.5))

    parts = [p for p in parts if p is not None]
    if not parts: 
        return pd.DataFrame(columns=["_grid_id","score","arrests_count"])

    merged = parts[0][["_grid_id","weighted","count"]].rename(columns={"count":"count0"})
    for i, p in enumerate(parts[1:], start=1):
        merged = merged.merge(p[["_grid_id","weighted","count"]].rename(columns={"count":f"count{i}"}), on="_grid_id", how="outer")
        merged["weighted"] = merged["weighted"].fillna(0) + merged.pop("weighted_y").fillna(0) if "weighted_y" in merged else merged["weighted"].fillna(0)
        if "weighted_x" in merged:
            merged["weighted"] = merged.pop("weighted_x") + merged["weighted"]

    # arrests tally: if we had an arrests_df piece, capture its counts
    arrests_piece = piece(arrests_df, 2.0) if arrests_df is not None else None
    merged["score"] = merged["weighted"].fillna(0)
    if arrests_piece is not None:
        merged = merged.merge(arrests_piece[["_grid_id","count"]].rename(columns={"count":"arrests_count"}), on="_grid_id", how="left")
    else:
        merged["arrests_count"] = 0

    return merged.fillna(0)

def build_service_index(service_df):
    """Return a mapping grid_id -> {zip, names, top_types} and a point table for markers."""
    if service_df is None or service_df.empty:
        return {}, pd.DataFrame()
    lat = detect_col(service_df, ["lat","latitude","y"])
    lon = detect_col(service_df, ["lon","lng","longitude","x"])
    if not lat or not lon:
        return {}, pd.DataFrame()
    svc = filter_bbox(service_df, lat, lon)
    if svc.empty:
        return {}, pd.DataFrame()
    svc = round_coord(svc, lat, lon, prec=3)
    name_col = detect_col(svc, ["name","business","venue","requester"])
    type_col = detect_col(svc, ["request","type","topic","category","service"])
    zip_series = inferred_zip(svc)
    svc = svc.assign(_zip=zip_series.values)
    # Build lookup
    idx = {}
    for gid, chunk in svc.groupby("_grid_id"):
        names = []
        if name_col:
            names = chunk[name_col].astype(str).fillna("").tolist()
            names = [n for n in names if n and n.lower() != "nan"]
            names = list(pd.unique(names))[:8]
        top_types = []
        if type_col:
            top_types = chunk[type_col].astype(str).value_counts().head(5).index.tolist()
        zip_mode = "75226"
        if "_zip" in chunk:
            try:
                zip_mode = str(chunk["_zip"].mode().iloc[0])
            except Exception:
                pass
        # store sample point (for marker coords)
        lat0 = chunk["_lat_r"].iloc[0]
        lon0 = chunk["_lon_r"].iloc[0]
        idx[gid] = {
            "zip": zip_mode,
            "names": names,
            "top_types": top_types,
            "lat": float(lat0),
            "lon": float(lon0),
            "count": int(len(chunk))
        }
    # Return also a point dataframe for markers
    pts = pd.DataFrame.from_records(
        [{"_grid_id":gid, **v} for gid, v in idx.items()]
    )
    return idx, pts

def legend_html():
    # Simple 3-band legend matching low/med/high heat
    return """
    <div class="legend">
      <h4>Crime Score (heat)</h4>
      <div class="row"><span class="swatch" style="background:#3B82F6"></span> Low</div>
      <div class="row"><span class="swatch" style="background:#F59E0B"></span> Medium</div>
      <div class="row"><span class="swatch" style="background:#EF4444"></span> High</div>
    </div>
    """

# ------------------ Load all CSVs ------------------
datasets = load_all_csvs(DATA_DIR)
nonempty = {k:v for k,v in datasets.items() if not v.empty}
if not nonempty:
    st.warning("⚠️ No CSVs found in `data/`.")
    st.stop()

st.markdown('<div class="gold"><b>🌟 Real Display:</b> Using your local CSVs (no external calls)</div>', unsafe_allow_html=True)

# ------------------ Identify key datasets ------------------
def pick_df(by_keywords):
    for name, df in nonempty.items():
        if any(kw in name.lower() for kw in by_keywords):
            return df
    # fallback: try content-based by presence of likely columns
    for df in nonempty.values():
        if any(detect_col(df,[k]) for k in ["offense","crime","arrest"]) and "crime" in by_keywords:
            return df
    return None

visits_df  = pick_df(["visit","foot","weekly"])
service_df = pick_df(["311","service","request"])
crime_df   = pick_df(["crime","police"])
arrests_df = pick_df(["arrest"])

# ------------------ Predict avg foot traffic ------------------
pred_avg_visits = predict_avg_foot_traffic(visits_df)

# ------------------ Build crime score grid ------------------
score_grid = build_crime_score(crime_df, arrests_df, service_df)  # _grid_id, score, arrests_count

# ------------------ Build service lookup & marker points ------------------
svc_lookup, svc_points = build_service_index(service_df)

# ------------------ Controls ------------------
st.sidebar.header("Map Controls")
radius = st.sidebar.slider("Heat radius", 8, 40, 22)
blur   = st.sidebar.slider("Heat blur", 6, 40, 18)
min_op = st.sidebar.slider("Min opacity", 0.0, 1.0, 0.35)

# ------------------ Tabs ------------------
tab_map, tab_overview, tab_tables = st.tabs(["🗺 All-in-One Map", "📊 Overview", "🔎 Data Tables"])

# ------------------ Map Tab ------------------
with tab_map:
    st.subheader("Deep Ellum Heatmap with Predictive Popups")

    # Build heat points from score grid by converting grid_id back to lat/lon
    heat_points = []
    if not score_grid.empty:
        for _, r in score_grid.iterrows():
            try:
                lat_r, lon_r = map(float, r["_grid_id"].split(","))
                weight = float(max(r["score"], 0.1))
                heat_points.append([lat_r, lon_r, weight])
            except Exception:
                pass

    m = Map(location=[32.783, -96.783], zoom_start=15, control_scale=True, prefer_canvas=True)

    # Heat layer
    if heat_points:
        HeatMap(heat_points, radius=radius, blur=blur, min_opacity=min_op, max_zoom=18).add_to(m)

    # Marker layer (clickable dots)
    marker_layer = FeatureGroup(name="Service Points").add_to(m)
    cluster = MarkerCluster(name="Locations", show=True).add_to(marker_layer)

    # Build popups from svc_points and join with score/arrests
    if not svc_points.empty:
        joined = svc_points.merge(score_grid[["_grid_id","score","arrests_count"]], on="_grid_id", how="left").fillna(0)
        for _, row in joined.iterrows():
            gid = row["_grid_id"]
            lat, lon = float(row["lat"]), float(row["lon"])
            score = int(row["score"])
            arrests = int(row["arrests_count"])
            svc = svc_lookup.get(gid, {})
            names = svc.get("names", [])
            top_types = svc.get("top_types", [])
            zipc = svc.get("zip", "75226")
            count_here = svc.get("count", 0)

            # Predictive summary (medium popup)
            popup_html = f"""
            <div style="min-width: 280px; max-width: 360px; font-family:'Times New Roman',serif;">
              <div style="font-weight:700; font-size:16px; color:#1a365d; margin-bottom:4px;">
                Location • {lat:.3f}, {lon:.3f}
              </div>
              <div style="font-size:13px; color:#4a5568; margin-bottom:8px;">
                ZIP: <b>{zipc}</b> • Service records here: <b>{count_here}</b>
              </div>

              <div style="margin-bottom:8px;">
                <div style="font-weight:700;">Top Service Requests</div>
                <div style="font-size:13px;">{', '.join(top_types) if top_types else 'N/A'}</div>
              </div>

              <div style="margin-bottom:8px;">
                <div style="font-weight:700;">Related Names</div>
                <div style="font-size:13px;">{', '.join(names) if names else 'N/A'}</div>
              </div>

              <div style="margin-bottom:8px;">
                <div style="font-weight:700;">Crime Score (local)</div>
                <div style="font-size:13px;">{score} (heat influences color)</div>
              </div>

              <div style="margin-bottom:8px;">
                <div style="font-weight:700;">Arrests (local tally)</div>
                <div style="font-size:13px;">{arrests}</div>
              </div>

              <div style="margin-bottom:2px;">
                <div style="font-weight:700;">Avg Foot Traffic (pred.)</div>
                <div style="font-size:13px;">≈ {int(pred_avg_visits):,} / day</div>
              </div>
            </div>
            """
            CircleMarker(
                location=(lat, lon),
                radius=5,
                color="#1d4ed8",
                fill=True,
                fill_opacity=0.9,
                popup=popup_html
            ).add_to(cluster)

    LayerControl(position="topleft").add_to(m)
    out = st_folium(m, height=680, width=1200)

    # Legend
    st.markdown(legend_html(), unsafe_allow_html=True)

# ------------------ Overview Tab ------------------
with tab_overview:
    st.subheader("At-a-Glance")
    cols = st.columns(4)
    with cols[0]:
        total_points = int(len(svc_points)) if not svc_points.empty else 0
        st.markdown(f'<div class="card"><b>Service locations</b><br><span style="font-size:1.5rem;">{total_points:,}</span></div>', unsafe_allow_html=True)
    with cols[1]:
        total_arrests = int(score_grid["arrests_count"].sum()) if not score_grid.empty else 0
        st.markdown(f'<div class="card"><b>Arrests tally</b><br><span style="font-size:1.5rem;">{total_arrests:,}</span></div>', unsafe_allow_html=True)
    with cols[2]:
        total_score = int(score_grid["score"].sum()) if not score_grid.empty else 0
        st.markdown(f'<div class="card"><b>Total crime score</b><br><span style="font-size:1.5rem;">{total_score:,}</span></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="card"><b>Avg foot traffic (pred.)</b><br><span style="font-size:1.5rem;">{int(pred_avg_visits):,}/day</span></div>', unsafe_allow_html=True)

    # Optional quick trend if visits present
    if visits_df is not None and not visits_df.empty:
        dc = detect_col(visits_df, ["date","week","datetime","timestamp"])
        vc = detect_col(visits_df, ["visits","count","foot","volume"])
        if dc and vc:
            tmp = visits_df[[dc,vc]].copy()
            tmp[dc] = pd.to_datetime(tmp[dc], errors="coerce")
            tmp = tmp.dropna()
            if not tmp.empty:
                agg = tmp.groupby(tmp[dc].dt.date)[vc].sum().reset_index(name="value")
                fig = px.line(agg, x=agg.columns[0], y="value", title="Foot Traffic (total)")
                st.plotly_chart(fig, use_container_width=True)

# ------------------ Data Tables Tab ------------------
with tab_tables:
    st.subheader("Loaded Datasets")
    for name, df in datasets.items():
        if df is not None and not df.empty:
            st.write(f"**{name}** — rows: {len(df):,}")
            st.dataframe(df.head(300), use_container_width=True)

st.divider()
st.caption("KrowdGuide — Deep Ellum Focus • All-in-One Heatmap & Predictive Popups")

