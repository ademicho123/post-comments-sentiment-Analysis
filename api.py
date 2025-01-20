from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

app = FastAPI()

# Create a Pydantic model for the request
class CommentInput(BaseModel):
    comment: str

# Load the trained model
try:
    model = joblib.load("models/comment_sentiments_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    logger.info("Model and vectorizer loaded successfully")
    
    # Debug: Print model information
    logger.debug(f"Model type: {type(model)}")
    logger.debug(f"Model classes: {model.classes_ if hasattr(model, 'classes_') else 'No classes found'}")
except Exception as e:
    logger.error(f"Error loading model or vectorizer: {e}")
    raise RuntimeError("Failed to load model or vectorizer")

def preprocess_text(text: str) -> str:
    """Preprocess the input text."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    logger.debug(f"Preprocessed text: {text}")
    return text

def get_sentiment_scores(text: str) -> float:
    """Get VADER sentiment scores for the text."""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    logger.debug(f"VADER scores: {scores}")
    return scores['compound']

@app.post("/predict/")
async def predict(input_data: CommentInput) -> Dict:
    try:
        # Extract the comment
        comment = input_data.comment
        logger.debug(f"Received comment: {comment}")
        
        # Preprocess the comment
        processed_comment = preprocess_text(comment)
        
        # Get VADER sentiment score
        sentiment_score = get_sentiment_scores(comment)
        logger.debug(f"VADER sentiment score: {sentiment_score}")
        
        # Vectorize the processed comment
        comment_vector = vectorizer.transform([processed_comment])
        logger.debug(f"Vectorized shape: {comment_vector.shape}")
        
        # Make prediction
        raw_prediction = model.predict(comment_vector)[0]
        probabilities = model.predict_proba(comment_vector)[0]
        logger.debug(f"Raw prediction: {raw_prediction}")
        logger.debug(f"Prediction probabilities: {probabilities}")
        
        # Map numerical predictions to sentiment labels
        sentiment_map = {
            2: "negative",
            1: "neutral",
            0: "positive"
        }
        predicted_sentiment = sentiment_map[raw_prediction]
        
        # Calculate VADER-based label
        if sentiment_score < 0.0:
            expected_class = 2  # negative
        elif sentiment_score >= 0.4:
            expected_class = 0  # positive
        else:
            expected_class = 1  # neutral
            
        logger.debug(f"Expected class based on VADER: {expected_class}")
        
        return {
            "status": "success",
            "prediction": predicted_sentiment,
            "raw_prediction": int(raw_prediction),
            "confidence": float(max(probabilities)),
            "sentiment_score": float(sentiment_score),
            "probabilities": {f"class_{i}": float(p) for i, p in enumerate(probabilities)},
            "processed_text": processed_comment
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}