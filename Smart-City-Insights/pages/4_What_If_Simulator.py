
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="What-If Simulator", page_icon="🔄")

st.title("🔄 Policy Impact Simulator (What-If Analysis)")

# Load model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model not found.")
    st.stop()

st.markdown("Adjust the sliders to simulate policy changes (e.g., planting more trees or reducing traffic).")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Baseline Scenario")
    base_green = st.slider("Current Green Cover (%)", 0, 100, 20, key="base_green")
    base_traffic = st.slider("Current Traffic Density", 0, 100, 80, key="base_traffic")
    
    # Assumptions for other fixed vars
    fixed_energy = 200
    fixed_aqi = 150 
    fixed_pollution = 60
    
    base_input = pd.DataFrame({
        "Energy_Consumption": [fixed_energy],
        "AQI": [fixed_aqi],
        "Pollution_Level": [fixed_pollution],
        "Green_Cover": [base_green],
        "Traffic_Density": [base_traffic]
    })
    
    base_pred = model.predict(base_input)[0]
    st.metric("Baseline Sustainability Index", f"{base_pred:.2f}")

with col2:
    st.subheader("Simulated Scenario")
    sim_green = st.slider("Target Green Cover (%)", 0, 100, 40, key="sim_green")
    sim_traffic = st.slider("Target Traffic Density", 0, 100, 50, key="sim_traffic")
    
    # Simple logic: Improving traffic/green cover also improves AQI/Pollution slightly in simulation
    improved_aqi = fixed_aqi - (sim_green - base_green)*0.5 - (base_traffic - sim_traffic)*0.5
    improved_pollution = fixed_pollution - (sim_green - base_green)*0.3 - (base_traffic - sim_traffic)*0.3
    
    sim_input = pd.DataFrame({
        "Energy_Consumption": [fixed_energy],
        "AQI": [improved_aqi],
        "Pollution_Level": [improved_pollution],
        "Green_Cover": [sim_green],
        "Traffic_Density": [sim_traffic]
    })
    
    sim_pred = model.predict(sim_input)[0]
    delta = sim_pred - base_pred
    st.metric("Simulated Sustainability Index", f"{sim_pred:.2f}", delta=f"{delta:.2f}")

# Visual Comparison
fig = go.Figure(data=[
    go.Bar(name='Baseline', x=['Sustainability Index'], y=[base_pred]),
    go.Bar(name='Simulated', x=['Sustainability Index'], y=[sim_pred])
])
fig.update_layout(barmode='group', title="Impact Comparison")
st.plotly_chart(fig)
