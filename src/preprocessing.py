import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load raw data from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Perform data preprocessing, such as handling missing values and scaling.
    """
    # Handle missing values
    data.fillna(data.median(), inplace=True)
    
    # Example feature selection (remove unnecessary columns)
    if "ID" in data.columns:
        data.drop("ID", axis=1, inplace=True)
    
    # Separate features and target
    X = data.drop("target", axis=1)  # Replace 'target' with your column name
    y = data["target"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
