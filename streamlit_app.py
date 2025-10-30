import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 🧭 Page setup
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
            st.error(f"❌ Could not find 'DeepEllumVisits.csv' in either:\n{relative_path}\n{absolute_path}")
            return pd.DataFrame()

        df = pd.read_csv(file_path)
        return df

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()  # fallback

df = load_data()

if df.empty:
    st.warning("⚠️ No data loaded. Please check that 'DeepEllumVisits.csv' exists in the 'data' folder.")
    st.stop()

# 🧭 Detect possible venue column automatically
possible_venue_cols = [col for col in df.columns if "venue" in col.lower()]
venue_col = possible_venue_cols[0] if possible_venue_cols else None

if not venue_col:
    st.error("❌ Could not find a column related to venue in your data.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# 🧭 Sidebar Filters
venues = sorted(df[venue_col].dropna().unique().tolist())
weeks_col = "week_start_iso" if "week_start_iso" in df.columns else None

if not weeks_col:
