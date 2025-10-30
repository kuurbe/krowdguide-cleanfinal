import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Krowd Guide: Deep Ellum Foot Traffic", layout="wide")
st.title("📍 Deep Ellum Venue Foot Traffic Dashboard")

# 📂 Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/Trees_vs_DadaDallas_Sep2025.csv")
    return df

df = load_data()

# 🧭 Sidebar Filters
venues = sorted(df["venue"].unique())
weeks = sorted(df["week_start_iso"].unique())

selected_venue = st.sidebar.selectbox("Select Venue", ["All"] + venues)
selected_week = st.sidebar.selectbox("Select Week", ["All"] + weeks)

# 🔄 Apply Filters
filtered_df = df.copy()
if selected_venue != "All":
    filtered_df = filtered_df[filtered_df["venue"] == selected_venue]
if selected_week != "All":
    filtered_df = filtered_df[filtered_df["week_start_iso"] == selected_week]

# 📈 Foot Traffic Trend
st.subheader("📈 Weekly Foot Traffic")
fig, ax = plt.subplots()
sns.lineplot(data=filtered_df, x="week_start_iso", y="visits", hue="venue", marker="o", ax=ax)
ax.set_title("Foot Traffic Over Time")
ax.set_ylabel("Visits")
ax.set_xlabel("Week Start")
plt.xticks(rotation=45)
st.pyplot(fig)

# 📄 Raw Data Viewer
with st.expander("📄 Show Raw Data"):
    st.dataframe(filtered_df)