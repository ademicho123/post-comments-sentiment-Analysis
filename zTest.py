# zTest.py
from src.preprocessing import prepare_dataset
import pandas as pd

# Create some sample data
data = pd.DataFrame({
    'processed_comments': ['This is a positive comment', 'This is a negative comment', 'This is a neutral comment'],
    'target': [1, 0, 1]
})

# Call the prepare_dataset function with the sample data
result = prepare_dataset(data)

# Check if result is not None
if result is not None:
    X_train, X_test, y_train, y_test, vectorizer = result
    # Print the output of the function
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
else:
    print("No data to process.")
