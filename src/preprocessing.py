import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import vstack, csr_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

# NLTK dependencies 
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

__all__ = ['prepare_dataset', 'preprocess_data']

# Function to preprocess individual text
def preprocess_data(df):
    """
    Preprocess the 'comments' column in the DataFrame, including text cleaning and NLP processing.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with a 'comments' column.
    
    Returns:
    pd.DataFrame: DataFrame with an additional 'processed_comments' column containing cleaned text.
    """
    # Ensure the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Ensure the 'comments' column exists
    if 'comments' not in df.columns:
        raise ValueError("The input DataFrame must have a 'comments' column.")

    # Initialize NLTK components
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def process_text(text):
        """
        Clean and preprocess a single comment.
        """
        # Convert to string if needed
        text = str(text)
        
        # Lowercase the text
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove emojis
        text = emoji.replace_emoji(text, '')
        
        # Remove emoticons
        text = re.sub(r'[:;=]-?[()DPp]', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', ' ', text)  # Punctuation
        text = re.sub(r'\d+', ' ', text)      # Numbers
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Tokenize, remove stop words, and lemmatize
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

        return ' '.join(tokens)

    # Apply preprocessing to the 'comments' column
    df['processed_comments'] = df['comments'].apply(process_text)

    return df

# Prepare Dataset Function
def prepare_dataset(data, target_size=5000, random_state=42):
    """
    Prepare dataset for training, increasing its size using SMOTE.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame with 'processed_comments' and 'target' columns
    target_size (int): Desired size of training dataset after SMOTE
    random_state (int): Random state for reproducibility
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, vectorizer)
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame")
    
    if 'processed_comments' not in data.columns or 'target' not in data.columns:
        raise ValueError("Data must contain 'processed_comments' and 'target' columns")
    
    # Extract features and target
    X = data['processed_comments'].copy()
    y = data['target'].copy()
    
    # Remove empty or invalid text
    print("\nCleaning data...")
    mask = X.apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 0)
    X = X[mask]
    y = y[mask]
    
    print(f"Initial dataset size: {len(y)}")
    print("Initial class distribution:")
    print(y.value_counts())
    
    # TF-IDF Vectorization
    print("\nPerforming TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)
    X = vectorizer.fit_transform(X)
    
    # Split the data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )
    
    # Initialize SMOTE
    print("\nApplying SMOTE to increase dataset size...")
    smote = SMOTE(
        sampling_strategy='auto',
        random_state=random_state,
        k_neighbors=min(5, len(y_train) - 1)
    )
    
    current_size = len(y_train)
    iterations = 0
    max_iterations = 10
    
    while current_size < target_size and iterations < max_iterations:
        print(f"Iteration {iterations + 1}: Current size = {current_size}")
        
        try:
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            if len(y_resampled) > target_size:
                indices = np.random.RandomState(random_state).choice(
                    len(y_resampled),
                    target_size,
                    replace=False
                )
                X_resampled = X_resampled[indices]
                y_resampled = y_resampled[indices]
                break
            
            X_train = X_resampled
            y_train = y_resampled
            current_size = len(y_train)
            iterations += 1
            
        except ValueError as e:
            print(f"Warning: SMOTE failed on iteration {iterations + 1}. Error: {str(e)}")
            break
    
    # Convert y_train to pandas Series
    y_train = pd.Series(y_train)
    
    # Final shuffle
    print("\nShuffling final dataset...")
    indices = np.random.RandomState(random_state).permutation(len(y_train))
    X_train = X_train[indices]
    y_train = y_train.iloc[indices].reset_index(drop=True)
    
    print("\nFinal dataset statistics:")
    print(f"Training set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")
    print("\nFinal class distribution in training set:")
    print(y_train.value_counts())
    
    return X_train, X_test, y_train, y_test, vectorizer



