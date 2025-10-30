# KrowdGuide Smart City OS - All-in-one Prototype
# Run: streamlit run streamlit_app.py
# Place your CSVs in ./data/ (supports *.csv and *.csv.gz)

import os
from pathlib import Path
from datetime import datetime
import io

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import humanize

# ------------------ App Config ------------------
st.set_page_config(page_title="KrowdGuide Smart City OS", layout="wide")
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

# ------------------ Named datasets (best-effort) ------------------
PATH_VISITS   = pick_by_name("DeepEllumVisits") or pick_by_name("WeeklyVisits")
PATH_ARRESTS  = pick_by_name("arrests")
PATH_SERVICE  = pick_by_name("service requests") or pick_by_name("311")
PATH_TRAFFIC  = pick_by_name("traffic") or pick_by_name("daily_clean")
PATH_BIKE_PED = pick_by_name("bike_pedestrian") or pick_by_name("bike") or pick_by_name("pedestrian")
PATH_WEATHER  = pick_by_name("weather")
PATH_TXDOT    = pick_by_name("TxDOT")
PATH_TREES    = pick_by_name("Trees")

# ------------------ DASHBOARD ------------------
if page == "Dashboard":
    st.subheader("📊 City Intelligence Overview")

    df_visits  = load_csv(PATH_VISITS) if PATH_VISITS else pd.DataFrame()
    df_arrests = load_csv(PATH_ARRESTS) if PATH_ARRESTS else pd.DataFrame()
    df_service = load_csv(PATH_SERVICE) if PATH_SERVICE else pd.DataFrame()
    df_traffic = load_csv(PATH_TRAFFIC) if PATH_TRAFFIC else pd.DataFrame()
    df_weather = load_csv(PATH_WEATHER) if PATH_WEATHER else pd.DataFrame()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        vcol = detect_col(df_visits, "visits", "count")
        last_visits = int(df_visits[vcol].iloc[-1]) if (not df_visits.empty and vcol) else 0
        st.metric("Weekly Visits (last row)", f"{last_visits:,}")
    with k2:
        st.metric("Arrests (rows)", f"{len(df_arrests):,}")
    with k3:
        st.metric("Service Requests (rows)", f"{len(df_service):,}")
    with k4:
        st.metric("Traffic Records (rows)", f"{len(df_traffic):,}")

    # Foot Traffic Trend
    if not df_visits.empty:
        wk = detect_col(df_visits, "week_start_iso", "week")
        vcol = detect_col(df_visits, "visits")
        venue = detect_col(df_visits, "venue")
        if wk and vcol:
            st.markdown("#### Foot Traffic Trend")
            fig, ax = figure()
            sns.lineplot(data=df_visits, x=wk, y=vcol, hue=venue, marker="o", ax=ax)
            ax.set_xlabel("Week"); ax.set_ylabel("Visits"); ax.set_title("Foot Traffic Over Time")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Top Service/Request Categories
    if not df_service.empty:
        st.markdown("#### Top Service/Request Categories")
        req = detect_col(df_service, "request", "service", "topic")
        if req:
            vc = df_service[req].astype("string").str.strip().fillna("⟂ NA").value_counts().head(10).reset_index()
            vc.columns = [req, "count"]
            fig, ax = figure()
            sns.barplot(data=vc, x="count", y=req, ax=ax)
            ax.set_title("Top Requests")
            st.pyplot(fig)

    # Weather Snapshot (optional)
    if not df_weather.empty:
        st.markdown("#### Weather Snapshot")
        tcol = detect_col(df_weather, "temperature", "temp", "temp_f", "temp_c")
        pcol = detect_col(df_weather, "precip", "rain", "precipitation")
        hcol = detect_col(df_weather, "humidity")
        dcol = detect_col(df_weather, "datetime", "timestamp", "date", "time", prefer_startswith=True)
        cols = st.columns(3)
        with cols[0]:
            if tcol: st.metric("Last Temperature", f"{df_weather[tcol].dropna().iloc[-1]:.1f}")
        with cols[1]:
            if pcol: st.metric("Last Precip", f"{df_weather[pcol].dropna().iloc[-1]:.2f}")
        with cols[2]:
            if hcol: st.metric("Last Humidity", f"{df_weather[hcol].dropna().iloc[-1]:.0f}%")

# ------------------ MOBILITY ------------------
elif page == "Mobility":
    st.subheader("🚶 Mobility: Traffic, Bike/Ped, Footfall")
    tabs = st.tabs(["Foot Traffic", "Traffic", "Bike/Ped"])

    # Foot Traffic
    with tabs[0]:
        if PATH_VISITS:
            df = load_csv(PATH_VISITS)
            st.caption(f"Dataset: {PATH_VISITS.name}")
            wk = detect_col(df, "week_start_iso", "week")
            vcol = detect_col(df, "visits")
            venue = detect_col(df, "venue")
            if wk and vcol:
                weeks = sorted(df[wk].dropna().unique().tolist())
                selected_week = st.selectbox("Filter by week", ["All"] + weeks)
                if venue:
                    venues = sorted(df[venue].dropna().unique().tolist())
                    selected_venue = st.selectbox("Filter by venue", ["All"] + venues)
                else:
                    selected_venue = "All"
                fdf = df.copy()
                if selected_week != "All":
                    fdf = fdf[fdf[wk] == selected_week]
                if venue and selected_venue != "All":
                    fdf = fdf[fdf[venue] == selected_venue]
                chart = st.radio("Chart", ["Line", "Bar"], horizontal=True, key="mob_visits_chart")
                fig, ax = figure()
                if chart == "Line":
                    sns.lineplot(data=fdf, x=wk, y=vcol, hue=venue, marker="o", ax=ax)
                else:
                    sns.barplot(data=fdf, x=wk, y=vcol, hue=venue, ax=ax)
                ax.set_title("Foot Traffic"); plt.xticks(rotation=45)
                st.pyplot(fig)
                preview_and_download(df, PATH_VISITS.name, key_suffix="visits")
            else:
                st.info("Could not detect `week`/`visits` columns.")
        else:
            st.warning("No visits dataset found.")

    # Traffic
    with tabs[1]:
        if PATH_TRAFFIC:
            df = load_csv(PATH_TRAFFIC)
            st.caption(f"Dataset: {PATH_TRAFFIC.name}")
            date_col = detect_col(df, "date", "day", prefer_startswith=True)
            vol_col  = detect_col(df, "volume", "count", "avg")
            if date_col and vol_col:
                chart = st.radio("Chart", ["Line", "Bar"], horizontal=True, key="traffic_chart")
                fig, ax = figure()
                if chart == "Line":
                    sns.lineplot(data=df, x=date_col, y=vol_col, marker="o", ax=ax)
                else:
                    sns.barplot(data=df, x=date_col, y=vol_col, ax=ax)
                ax.set_title("Traffic Volume"); plt.xticks(rotation=45)
                st.pyplot(fig)
            preview_and_download(df, PATH_TRAFFIC.name, key_suffix="traffic")
        else:
            st.warning("No traffic dataset found.")

    # Bike/Ped
    with tabs[2]:
        if PATH_BIKE_PED:
            df = load_csv(PATH_BIKE_PED)
            st.caption(f"Dataset: {PATH_BIKE_PED.name}")
            x = detect_col(df, "date", "day", prefer_startswith=True)
            y = detect_col(df, "count", "avg", "volume")
            if x and y:
                fig, ax = figure()
                sns.lineplot(data=df, x=x, y=y, marker="o", ax=ax)
                ax.set_title("Bike/Ped Counts"); plt.xticks(rotation=45)
                st.pyplot(fig)
            preview_and_download(df, PATH_BIKE_PED.name, key_suffix="bikeped")
        else:
            st.warning("No bike/ped dataset found.")

# ------------------ PUBLIC SAFETY ------------------
elif page == "Public Safety":
    st.subheader("👮 Public Safety: Arrests & Requests")
    tabs = st.tabs(["Arrests", "Service Requests"])

    with tabs[0]:
        if PATH_ARRESTS:
            df = load_csv(PATH_ARRESTS)
            st.caption(f"Dataset: {PATH_ARRESTS.name}")
            cat = detect_col(df, "offense", "charge", "category")
            date_col = detect_col(df, "date", "arrest", "datetime", prefer_startswith=True)
            if cat:
                st.markdown("**Top Offenses**")
                vc = df[cat].astype("string").str.strip().fillna("⟂ NA").value_counts().head(15).reset_index()
                vc.columns = [cat, "count"]
                fig, ax = figure()
                sns.barplot(data=vc, x="count", y=cat, ax=ax)
                ax.set_title("Top Offenses"); st.pyplot(fig)
            if date_col:
                st.markdown("**Arrests Over Time**")
                ts = df.groupby(date_col).size().reset_index(name="count")
                fig, ax = figure()
                sns.lineplot(data=ts, x=date_col, y="count", marker="o", ax=ax)
                plt.xticks(rotation=45); ax.set_title("Arrests Over Time")
                st.pyplot(fig)
            preview_and_download(df, PATH_ARRESTS.name, key_suffix="arrests")
        else:
            st.warning("No arrests dataset found.")

    with tabs[1]:
        if PATH_SERVICE:
            df = load_csv(PATH_SERVICE)
            st.caption(f"Dataset: {PATH_SERVICE.name}")
            req = detect_col(df, "request", "service", "topic")
            date_col = detect_col(df, "date", "week", "datetime", prefer_startswith=True)
            if req:
                st.markdown("**Top Request/Service Categories**")
                vc = df[req].astype("string").str.strip().fillna("⟂ NA").value_counts().head(15).reset_index()
                vc.columns = [req, "count"]
                fig, ax = figure()
                sns.barplot(data=vc, x="count", y=req, ax=ax)
                ax.set_title("Requests Top Categories"); st.pyplot(fig)
            if date_col and req:
                st.markdown("**Requests Trend**")
                trend = (df.assign(__cat=df[req].astype("string").str.strip().fillna("⟂ NA"))
                           .groupby([date_col, "__cat"]).size().reset_index(name="count"))
                fig, ax = figure()
                sns.lineplot(data=trend, x=date_col, y="count", hue="__cat", marker="o", ax=ax)
                plt.xticks(rotation=45); ax.set_title("Requests Over Time")
                st.pyplot(fig)
            preview_and_download(df, PATH_SERVICE.name, key_suffix="service")
        else:
            st.warning("No service-requests dataset found.")

# ------------------ INFRASTRUCTURE ------------------
elif page == "Infrastructure":
    st.subheader("🛠️ Infrastructure: 311 & TxDOT")
    tabs = st.tabs(["311 Calls", "TxDOT / Roadwork"])

    with tabs[0]:
        p = PATH_SERVICE or pick_by_name("311")
        if p:
            df = load_csv(p)
            st.caption(f"Dataset: {p.name}")
            topic = detect_col(df, "request", "service", "topic")
            status = detect_col(df, "status", "state")
            if topic:
                st.markdown("**311 Topics**")
                vc = df[topic].astype("string").fillna("⟂ NA").value_counts().head(15).reset_index()
                vc.columns = [topic, "count"]
                fig, ax = figure()
                sns.barplot(data=vc, x="count", y=topic, ax=ax)
                ax.set_title("Top 311 Topics"); st.pyplot(fig)
            if status:
                st.markdown("**Status Distribution**")
                vc = df[status].astype("string").fillna("⟂ NA").value_counts().reset_index()
                vc.columns = [status, "count"]
                fig, ax = figure()
                sns.barplot(data=vc, x="count", y=status, ax=ax)
                ax.set_title("311 Status"); st.pyplot(fig)
            preview_and_download(df, p.name, key_suffix="infra311")
        else:
            st.warning("No 311 dataset found.")

    with tabs[1]:
        if PATH_TXDOT:
            df = load_csv(PATH_TXDOT)
            st.caption(f"Dataset: {PATH_TXDOT.name}")
            date_col = detect_col(df, "date", "week", "datetime", prefer_startswith=True)
            metric = detect_col(df, "count", "closures", "incidents", "events")
            if date_col and metric:
                fig, ax = figure()
                sns.lineplot(data=df, x=date_col, y=metric, marker="o", ax=ax)
                plt.xticks(rotation=45); ax.set_title("TxDOT / Roadwork")
                st.pyplot(fig)
            preview_and_download(df, PATH_TXDOT.name, key_suffix="txdot")
        else:
            st.warning("No TxDOT dataset found.")

# ------------------ SUSTAINABILITY (incl. WEATHER) ------------------
elif page == "Sustainability":
    st.subheader("🌳 Sustainability & Weather Insights")
    tabs = st.tabs(["Environment / Trees", "Weather Trends"])

    # Trees & Green Data
    with tabs[0]:
        if PATH_TREES:
            df = load_csv(PATH_TREES)
            st.caption(f"Dataset: {PATH_TREES.name}")
            metric = detect_col(df, "tree", "species", "green", "park")
            count_col = detect_col(df, "count", "number", "qty")
            if metric:
                vc = df[metric].astype("string").fillna("⟂ NA").value_counts().head(20).reset_index()
                vc.columns = [metric, "count"]
                fig, ax = figure()
                sns.barplot(data=vc, x="count", y=metric, ax=ax)
                ax.set_title("Top Tree/Green Categories"); st.pyplot(fig)
            if count_col:
                st.write("Summary")
                st.dataframe(df[[metric, count_col]].describe(include="all") if metric else df[[count_col]].describe(), use_container_width=True)
            preview_and_download(df, PATH_TREES.name, key_suffix="trees")
        else:
            st.info("No tree/green dataset found.")

    # Weather Data
    with tabs[1]:
        if PATH_WEATHER:
            df = load_csv(PATH_WEATHER)
            st.caption(f"Dataset: {PATH_WEATHER.name}")

            # Flexible column detection
            dcol = detect_col(df, "datetime", "timestamp", "date", "time", prefer_startswith=True)
            tcol = detect_col(df, "temperature", "temp", "temp_f", "temp_c")
            pcol = detect_col(df, "precip", "rain", "precipitation")
            hcol = detect_col(df, "humidity", "rh")

            # Try to parse datetimes if needed
            if dcol and not np.issubdtype(df[dcol].dtype, np.datetime64):
                with st.spinner("Parsing datetimes..."):
                    try:
                        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
                    except Exception:
                        pass

            cols = st.columns(3)
            with cols[0]:
                if tcol and not df[tcol].dropna().empty:
                    st.metric("Last Temperature", f"{df[tcol].dropna().iloc[-1]:.1f}")
            with cols[1]:
                if pcol and not df[pcol].dropna().empty:
                    st.metric("Last Precip", f"{df[pcol].dropna().iloc[-1]:.2f}")
            with cols[2]:
                if hcol and not df[hcol].dropna().empty:
                    st.metric("Last Humidity", f"{df[hcol].dropna().iloc[-1]:.0f}%")

            chart = st.radio("Weather Chart", ["Temperature", "Precipitation", "Humidity"], horizontal=True)
            fig, ax = figure()
            if chart == "Temperature" and dcol and tcol:
                sns.lineplot(data=df, x=dcol, y=tcol, ax=ax)
                ax.set_title("Temperature Over Time")
            elif chart == "Precipitation" and dcol and pcol:
                sns.lineplot(data=df, x=dcol, y=pcol, ax=ax)
                ax.set_title("Precipitation Over Time")
            elif chart == "Humidity" and dcol and hcol:
                sns.lineplot(data=df, x=dcol, y=hcol, ax=ax)
                ax.set_title("Humidity Over Time")
            else:
                ax.text(0.5, 0.5, "Required columns not found", ha="center", va="center", transform=ax.transAxes)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            preview_and_download(df, PATH_WEATHER.name, key_suffix="weather")
        else:
            st.info("No weather dataset found.")

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
