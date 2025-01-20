from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
from typing import Dict

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

app = FastAPI()

# Create a Pydantic model for the request
class CommentInput(BaseModel):
    comment: str

# Load the trained model
try:
    model = joblib.load("models/comment_sentiments_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")  # You'll need to save this during training
    print("Model and vectorizer loaded successfully")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    raise RuntimeError("Failed to load model or vectorizer")

def preprocess_text(text: str) -> str:
    """Preprocess the input text."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def get_sentiment_scores(text: str) -> float:
    """Get VADER sentiment scores for the text."""
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

@app.post("/predict/")
async def predict(input_data: CommentInput) -> Dict:
    try:
        # Extract the comment
        comment = input_data.comment
        
        # Preprocess the comment
        processed_comment = preprocess_text(comment)
        
        # Get sentiment score
        sentiment_score = get_sentiment_scores(comment)
        
        # Vectorize the processed comment
        comment_vector = vectorizer.transform([processed_comment])
        
        # Make prediction
        prediction = model.predict(comment_vector)[0]
        probabilities = model.predict_proba(comment_vector)[0]
        
        # Map numerical predictions to sentiment labels
        sentiment_map = {0: "positive", 1: "neutral", 2: "negative"}
        predicted_sentiment = sentiment_map[prediction]
        
        return {
            "status": "success",
            "prediction": predicted_sentiment,
            "confidence": float(max(probabilities)),
            "sentiment_score": float(sentiment_score),
            "processed_text": processed_comment
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}