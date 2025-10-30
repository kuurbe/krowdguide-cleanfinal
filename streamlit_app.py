import os
from pathlib import Path
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 🧩 Render / Streamlit Port Config
port = int(os.environ.get("PORT", 8501))
st.set_page_config(page_title="Krowd Guide: Deep Ellum Foot Traffic", layout="wide")
st.title("📍 Deep Ellum Venue Foot Traffic Dashboard")

# 📂 Load Data
@st.cache_data
def load_data():
    try:
        base_dir = Path(__file__).resolve().parent
        relative_path = base_dir / "data" / "DeepEllumVisits.csv"
        absolute_path = Path(r"C:\Users\juhco\OneDrive\Documents\krowdguide-cleanfinal\data\DeepEllumVisits.csv")

        # ✅ Try relative path first, then absolute fallback
        if relative_path.exists():
            file_path = relative_path
        elif absolute_path.exists():
            file_path = absolute_path
        else:
            return pd.DataFrame()  # return empty to trigger uploader

        df = pd.read_csv(file_path)
        return df

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()  # fallback

df = load_data()

# 🧾 Handle missing data with upload fallback
if df.empty:
    st.warning("⚠️ No data loaded. Please upload your DeepEllumVisits.csv file below.")
    uploaded = st.file_uploader("Upload DeepEllumVisits.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("✅ File uploaded successfully!")
    else:
        st.stop()

# 🧭 Detect possible venue column automatically
possible_venue_cols = [col for col in df.columns if "venue" in col.lower()]
venue_col = possible_venue_cols[0] if possible_venue_cols else None

if not venue_col:
    st.error("❌ Could not find a column related to venue in your data.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# 🧭 Detect week column
weeks_col = "week_start_iso" if "week_start_iso" in df.columns else None

if not weeks_col:
    st.error("❌ Column 'week_start_iso' not found in data.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# 🧭 Sidebar Filters
venues = sorted(df[venue_col].dropna().unique().tolist())
weeks = sorted(df[weeks_col].dropna().unique().tolist())

selected_venue = st.sidebar.selectbox("Select Venue", ["All"] + venues)
selected_week = st.sidebar.selectbox("Select Week", ["All"] + weeks)

# 🔄 Apply Filters
filtered_df = df.copy()
if selected_venue != "All":
    filtered_df = filtered_df[filtered_df[venue_col] == selected_venue]
if selected_week != "All":
    filtered_df = filtered_df[filtered_df[weeks_col] == selected_week]

# 📊 Chart Type Toggle
st.subheader("📈 Weekly Foot Traffic")

if "visits" not in filtered_df.columns:
    st.error("❌ Missing 'visits' column in data.")
else:
    col1, col2 = st.columns([3, 1])
    with col2:
        chart_type = st.radio("Chart Type", ["Line", "Bar"], horizontal=False)

    with col1:
        fig, ax = plt.subplots()
        if chart_type == "Line":
            sns.lineplot(
                data=filtered_df,
                x=weeks_col,
                y="visits",
                hue=venue_col,
                marker="o",
                ax=ax
            )
        else:
            sns.barplot(
                data=filtered_df,
                x=weeks_col,
                y="visits",
                hue=venue_col,
                ax=ax
            )

        ax.set_title("Foot Traffic Over Time")
        ax.set_ylabel("Visits")
        ax.set_xlabel("Week Start")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# 📄 Raw Data Viewer
with st.expander("📄 Show Raw Data"):
    st.dataframe(filtered_df)

# 💾 Optional: Download filtered data
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Filtered Data as CSV", data=csv, file_name="Filtered_DeepEllumVisits.csv", mime="text/csv")
