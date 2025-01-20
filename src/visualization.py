import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix for the sentiment analysis results.
    
    Parameters:
    y_true: True labels
    y_pred: Predicted labels
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

def plot_feature_importance(model, feature_names, X_test):
    """
    Create and save feature importance visualization for XGBoost model.
    
    Parameters:
    model: Trained XGBoost model
    feature_names: List of feature names (from TF-IDF vectorizer).
    """
    # Get feature importance scores
    importance_scores = model.feature_importances_

    # Create a feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })

    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()