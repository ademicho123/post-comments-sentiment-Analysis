from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load the trained model
model_path = "models/comment_sentiments_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

@app.post("/predict/")
async def predict(data: dict):
    try:
        # Convert JSON to DataFrame
        df = pd.DataFrame([data])
        
        # Add preprocessing if required
        # Example: df = preprocess_data(df)
        
        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]
        
        return {"prediction": int(prediction[0]), "probability": float(probability[0])}
    except Exception as e:
        return {"error": str(e)}
