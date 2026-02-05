
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_data

st.set_page_config(page_title="Data Dashboard", page_icon="📊", layout="wide")

st.title("📊 Urban Data Dashboard")

df = load_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Energy Consumption Trends")
    fig_energy = px.histogram(df, x="Energy_Consumption", nbins=30, title="Distribution of Energy Consumption")
    st.plotly_chart(fig_energy, use_container_width=True)

with col2:
    st.subheader("AQI Distribution")
    fig_aqi = px.box(df, y="AQI", title="Air Quality Index Spread")
    st.plotly_chart(fig_aqi, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Pollution vs Traffic")
    fig_scatter = px.scatter(df, x="Traffic_Density", y="Pollution_Level", color="Sustainability_Rating", title="Pollution vs Traffic Density")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
