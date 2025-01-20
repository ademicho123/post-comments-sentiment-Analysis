from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

def train_model(X_train, y_train):
    """
    Train an XGBoost model on the training data with hyperparameter tuning
    and class balancing.
    """
    # Create pipeline with SMOTE and XGBoost
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, 
                             eval_metric="mlogloss",
                             objective='multi:softprob',
                             random_state=42))
    ])

    # Enhanced parameter grid for sentiment analysis
    param_grid = {
        'xgb__n_estimators': [100, 200, 300],
        'xgb__max_depth': [3, 4, 5],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__subsample': [0.8, 0.9, 1.0],
        'xgb__min_child_weight': [1, 3],
        'xgb__colsample_bytree': [0.8, 1.0]
    }

    # GridSearchCV with stratification
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
        refit='f1_macro',  # Use F1 score for selecting best model
        cv=5,  # Increased from 3 to 5 for better validation
        verbose=2,
        n_jobs=-1
    )

    # Fit model
    print("Training model with hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    # Print detailed results
    print("\nBest Parameters:", grid_search.best_params_)
    print("\nBest Cross-Validation Scores:")
    for metric, score in grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_:grid_search.best_index_+1].items():
        print(f"{metric}: {score:.4f}")
    
    # Get class distribution before and after SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print("\nClass distribution before SMOTE:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Class distribution after SMOTE:", dict(zip(*np.unique(y_resampled, return_counts=True))))

    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model with detailed metrics and analysis.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Get classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix with labels
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Positive (0)', 'Neutral (1)', 'Negative (2)']
    
    print("\nConfusion Matrix:")
    print("True labels (rows) vs Predicted labels (columns)")
    print("\t\t" + "\t".join(labels))
    for i, row in enumerate(cm):
        print(f"{labels[i]}\t{row}")

    # Calculate and print prediction probabilities statistics
    print("\nPrediction Probability Statistics:")
    proba_df = pd.DataFrame(y_prob, columns=['Positive', 'Neutral', 'Negative'])
    print(proba_df.describe())

    return {
        'predictions': y_pred,
        'probabilities': y_prob,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def save_model(model, file_path):
    """
    Save the trained model to a file.
    """
    with open(file_path, "wb") as file:
        pickle.dump(model, file)

def load_model(file_path):
    """
    Load a trained model from a file.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)