# Prepare Dataset Function
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
def prepare_dataset(data, sample_frac=0.1, random_state=42):
    data = data.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    
    X = data['processed_comments']
    y = data['target']
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Define resampling strategy
    over = SMOTE(sampling_strategy='auto', random_state=random_state)
    under = RandomUnderSampler(sampling_strategy='auto', random_state=random_state)
    
    # Create a pipeline with SMOTE and RandomUnderSampler
    resampling = Pipeline([('over', over), ('under', under)])
    
    # Apply resampling
    X_train_resampled, y_train_resampled = resampling.fit_resample(X_train, y_train)
    
    print(f"Dataset prepared with train size: {X_train_resampled.shape[0]} and test size: {X_test.shape[0]}")
    return X_train_resampled, X_test, y_train_resampled, y_test, vectorizer