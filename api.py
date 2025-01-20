from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

# Load the trained model
model_path = "models/comment_sentiments_model.pkl"
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except pickle.UnpicklingError as e:
    print(f"Error loading model: {e}")
    # Try to load the model using joblib
    import joblib
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model using joblib: {e}")
        model = None

if model is None:
    print("Failed to load model")
else:
    print("Model loaded successfully")

@app.post("/predict/")
async def predict(data: dict):
    try:
        # Convert JSON to DataFrame
        df = pd.DataFrame([data])
        
        # Convert DataFrame to numpy array
        X = df.values
        
        # Make prediction
        prediction = model.predict(X)
        probability = model.predict_proba(X)[:, 1]
        
        return {"prediction": int(prediction[0]), "probability": float(probability[0])}
    except Exception as e:
        return {"error": str(e)}
