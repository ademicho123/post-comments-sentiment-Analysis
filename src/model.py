import pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score

def train_model(X_train, y_train):
    """
    Train a model on the training data.
    """
    model = IsolationForest(random_state=42)
    model.fit(X_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return performance metrics.
    """
    predictions = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

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
