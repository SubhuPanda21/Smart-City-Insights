
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SHAP is heavy and might fail on install. We wrap it.
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="Explainable AI", page_icon="🧠")

st.title("🧠 Explainable AI (XAI)")

st.markdown("""
Understanding **why** the AI predicts a certain sustainability score is crucial for trust and transparency.
We use SHAP (SHapley Additive exPlanations) to visualize feature contributions.
""")

if not SHAP_AVAILABLE:
    st.warning("SHAP library is not installed. Showing placeholder visualization.")
else:
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
            
        st.subheader("Global Feature Importance")
        
        # We need some background data for SHAP
        # Generate small sample
        from utils import generate_data
        df = generate_data(100)
        X = df[["Energy_Consumption", "AQI", "Pollution_Level", "Green_Cover", "Traffic_Density"]]
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        st.markdown("### Summary Plot")
        st.markdown("This plot shows how each feature impacts the model output. Points to the right increase the sustainability score, points to the left decrease it.")
        
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error calculating SHAP values: {e}")

