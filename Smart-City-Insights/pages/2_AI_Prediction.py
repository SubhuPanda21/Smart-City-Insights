
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="AI Prediction", page_icon="🔮")

st.title("🔮 AI Sustainability Predictor")

# Load model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model not found. Please run training script first.")
    st.stop()

st.sidebar.header("Input Parameters")

energy = st.sidebar.slider("Energy Consumption (kWh)", 50.0, 400.0, 200.0)
aqi = st.sidebar.slider("AQI Level", 0.0, 500.0, 100.0)
pollution = st.sidebar.slider("Pollution Level", 0.0, 100.0, 50.0)
green_cover = st.sidebar.slider("Green Cover (%)", 0.0, 100.0, 30.0)
traffic = st.sidebar.slider("Traffic Density", 0.0, 100.0, 50.0)

input_data = pd.DataFrame({
    "Energy_Consumption": [energy],
    "AQI": [aqi],
    "Pollution_Level": [pollution],
    "Green_Cover": [green_cover],
    "Traffic_Density": [traffic]
})

if st.button("Predict Sustainability Index"):
    prediction = model.predict(input_data)[0]
    
    st.metric(label="Predicted Sustainability Index", value=f"{prediction:.2f}/100")
    
    if prediction > 75:
        st.success("Rating: High Sustainability 🌿")
    elif prediction > 40:
        st.warning("Rating: Moderate Sustainability ⚠️")
    else:
        st.error("Rating: Low Sustainability 🚨")

    st.subheader("Feature Contribution")
    # Simple feature importance from Random Forest
    importances = model.feature_importances_
    features = input_data.columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
    st.bar_chart(importance_df.set_index("Feature"))
