
import streamlit as st

st.set_page_config(page_title="AI Advisor", page_icon="💡")

st.title("💡 AI Smart City Advisor")

st.subheader("Current Status Assessment")

aqi = st.number_input("Current AQI", value=150)
energy = st.number_input("Daily Energy Usage (kWh)", value=250)
traffic = st.number_input("Traffic Density (cars/min)", value=60)

recommendations = []

# Logic for recommendations
if aqi > 200:
    st.error(f"Critical AQI Level: {aqi}")
    recommendations.append("🚨 **Urgent:** Issue smog alert to residents.")
    recommendations.append("🚗 Restrict heavy vehicle entry into the city center.")
    recommendations.append("🏭 Inspect nearby industrial zones for emission violations.")
elif aqi > 100:
    st.warning(f"Unhealthy AQI Level: {aqi}")
    recommendations.append("⚠️ Encourage public transport use.")
    recommendations.append("🌳 Plan for more vertical gardens in high-traffic zones.")
else:
    st.success(f"Good AQI Level: {aqi}")
    recommendations.append("✅ Maintain current green policies.")

if energy > 300:
    st.warning("High Energy Consumption detected.")
    recommendations.append("💡 Incentivize solar panel installation for residential areas.")
    recommendations.append("🏢 Audit public buildings for energy efficiency.")

if traffic > 80:
    st.warning("High Traffic Congestion.")
    recommendations.append("🚦 Optimize traffic light timing using AI.")
    recommendations.append("🚲 Expand bicycle lanes and pedestrian zones.")

st.markdown("### 📋 Recommendations")
if recommendations:
    for rec in recommendations:
        st.markdown(f"- {rec}")
else:
    st.info("System metrics are within optimal ranges. No urgent actions needed.")
