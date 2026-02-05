
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from utils import generate_data

def train():
    print("Generating training data...")
    df = generate_data(2000)
    
    # Features and Target
    # We want to predict Sustainability Index based on inputs
    X = df[["Energy_Consumption", "AQI", "Pollution_Level", "Green_Cover", "Traffic_Density"]]
    y = df["Sustainability_Index"]
    
    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Saving model to model.pkl...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("Done.")

if __name__ == "__main__":
    train()
