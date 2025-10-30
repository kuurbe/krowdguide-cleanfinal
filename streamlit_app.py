import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Krowd Guide Dashboard", layout="wide")
st.title("📍 Deep Ellum Foot Traffic Indicator")

# 📂 Load Data
@st.cache_data
def load_data():
    visits = pd.read_csv("data/DeepEllumVisits.csv")
    trees = pd.read_csv("data/Trees_vs_DadaDallas_Sep2025.csv")
    calls = pd.read_csv("data/311 serv calls_clean.csv")
    arrests = pd.read_csv("data/deep ellum based arrests_clean.csv")
    bikeped = pd.read_csv("data/bike_pedestrian traffic daily_clean.csv")
    return visits, trees, calls, arrests, bikeped

visits, trees, calls, arrests, bikeped = load_data()

# 🧭 Sidebar Filters
st.sidebar.header("🔍 Filter Dashboard")
zones = sorted(visits["Zone"].unique())
selected_zone = st.sidebar.selectbox("Select Zone", ["All"] + zones)
selected_week = st.sidebar.selectbox("Select Week", ["All"] + sorted(visits["Week"].unique()))

# 🔄 Apply Filters
if selected_zone != "All":
    visits = visits[visits["Zone"] == selected_zone]
    trees = trees[trees["Zone"] == selected_zone]
    calls = calls[calls["Zone"] == selected_zone]
    arrests = arrests[arrests["Zone"] == selected_zone]
    bikeped = bikeped[bikeped["Zone"] == selected_zone]

if selected_week != "All":
    visits = visits[visits["Week"] == selected_week]
    calls = calls[calls["Week"] == selected_week]
    arrests = arrests[arrests["Week"] == selected_week]

# 📈 Foot Traffic Trend
st.subheader("📈 Weekly Foot Traffic")
fig1, ax1 = plt.subplots()
sns.lineplot(data=visits, x="Week", y="Visits", marker="o", ax=ax1)
ax1.set_title("Foot Traffic Over Time")
st.pyplot(fig1)

# 🏢 Venue Count vs Visits
st.subheader("🏢 Venue Count vs Foot Traffic")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=trees, x="VenueCount", y="Visits", hue="Zone", ax=ax2)
ax2.set_title("Venue Density vs Foot Traffic")
st.pyplot(fig2)

# 🌳 Tree Coverage Impact
st.subheader("🌳 Tree Coverage vs Foot Traffic")
fig3, ax3 = plt.subplots()
sns.scatterplot(data=trees, x="TreeCoverage", y="Visits", hue="Zone", ax=ax3)
ax3.set_title("Environmental Impact on Foot Traffic")
st.pyplot(fig3)

# 🚨 311 Calls Breakdown
st.subheader("🚨 311 Service Calls")
call_summary = calls.groupby("CallType")["Count"].sum().reset_index()
st.bar_chart(call_summary.set_index("CallType"))

# 👮 Arrests Overview
st.subheader("👮 Arrests by Week")
fig4, ax4 = plt.subplots()
sns.barplot(data=arrests, x="Week", y="ArrestCount", ax=ax4)
ax4.set_title("Arrests Over Time")
st.pyplot(fig4)

# 🚶 Bike/Pedestrian Flow
st.subheader("🚶 Pedestrian Flow by Time")
if "Day" in bikeped.columns and "Time" in bikeped.columns:
    pivot = bikeped.pivot_table(index="Day", columns="Time", values="Count", aggfunc="sum").fillna(0)
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax5)
    ax5.set_title("Pedestrian Activity Heatmap")
    st.pyplot(fig5)

# 📄 Raw Data Toggle
with st.expander("📄 Show Raw Data"):
    st.write("Visits")
    st.dataframe(visits)
    st.write("Trees & Venues")
    st.dataframe(trees)
    st.write("311 Calls")
    st.dataframe(calls)
    st.write("Arrests")
    st.dataframe(arrests)
    st.write("Bike/Pedestrian")
    st.dataframe(bikeped)