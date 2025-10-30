import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Krowd Guide: Venue Foot Traffic", layout="wide")
st.title("📍 Deep Ellum Venue Foot Traffic")

# 📂 Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data/Trees_vs_DadaDallas_Sep2025.csv")

df = load_data()

# 🧭 Sidebar Filters
venues = sorted(df["venue_name"].unique())
selected_venue = st.sidebar.selectbox("Select Venue", ["All"] + venues)
selected_week = st.sidebar.selectbox("Select Week", ["All"] + sorted(df["week_start_iso"].unique()))

# 🔄 Apply Filters
if selected_venue != "All":
    df = df[df["venue_name"] == selected_venue]

if selected_week != "All":
    df = df[df["week_start_iso"] == selected_week]

# 📈 Foot Traffic Trend
st.subheader("📈 Weekly Foot Traffic")
fig1, ax1 = plt.subplots()
sns.lineplot(data=df, x="week_start_iso", y="visits", hue="venue_name", marker="o", ax=ax1)
ax1.set_title("Foot Traffic Over Time")
ax1.set_ylabel("Visits")
ax1.set_xlabel("Week Start")
st.pyplot(fig1)

# 📄 Raw Data
with st.expander("📄 Show Raw Data"):
    st.dataframe(df)