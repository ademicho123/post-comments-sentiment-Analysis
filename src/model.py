from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle

def train_model(X_train, y_train):
    """
    Train an XGBoost model on the training data with hyperparameter tuning.
    """
    # Define XGBoost model
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    # Parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    # Fit model
    print("Training model with hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.2f}")

    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nOverall Accuracy: {accuracy:.3f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Return the results
    return {
        'predictions': predictions,
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, predictions),
        'confusion_matrix': cm
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