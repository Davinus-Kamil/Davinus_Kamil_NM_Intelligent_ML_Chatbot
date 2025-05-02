import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from pathlib import Path
import os

# Define paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_PATH = BASE_DIR / 'customer_queries.csv'
MODEL_PATH = BASE_DIR / 'chatbot_model.joblib'

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """Clean and preprocess text data."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def train_model():
    """Train the chatbot model using the dataset."""
    try:
        # Load the dataset with explicit quoting
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
            
        df = pd.read_csv(DATA_PATH, quoting=1)  # QUOTE_ALL mode
        print(f"Loaded dataset with {len(df)} rows")
        print("\nDataset columns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        
        # Preprocess the queries
        df['processed_query'] = df['query'].apply(preprocess_text)
        print("\nPreprocessed all queries")
        
        # Create and train the model pipeline
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', MultinomialNB())
        ])
        
        # Train the model
        model.fit(df['processed_query'], df['response'])
        print("Model training completed")
        
        # Save the model
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved successfully at {MODEL_PATH}")
        print(f"Model file size: {MODEL_PATH.stat().st_size / 1024:.2f} KB")
        
    except Exception as e:
        print(f"Error during model training: {type(e).__name__}: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 