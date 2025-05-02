import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import string
import os
import nltk
from nltk.corpus import stopwords
from utils import get_model_path, get_vectorizer_path, get_data_path, check_files_exist

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    """Clean and preprocess text data."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

def train_model():
    """Train the chatbot model using the dataset."""
    try:
        print("Loading dataset...")
        # Load the dataset
        data_path = get_data_path()
        print(f"Reading data from: {data_path}")
        df = pd.read_csv(data_path)
        print("\nFirst 5 rows of customer_queries.csv:")
        print(df.head())
        print("\nDataset shape:", df.shape)
        
        print("\nPreprocessing text...")
        # Preprocess the queries
        df['processed_query'] = df['query'].apply(preprocess_text)
        
        print("\nVectorizing text...")
        # Vectorization
        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['processed_query'])
        y = df['intent']
        
        print("\nTraining model...")
        # Train the model
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X, y)
        
        # Evaluate the model
        y_pred = model.predict(X)
        print("\nModel Evaluation:")
        print(classification_report(y, y_pred))
        
        print("\nSaving model and vectorizer...")
        # Save the model and vectorizer
        model_path = get_model_path()
        vectorizer_path = get_vectorizer_path()
        
        print(f"Saving model to: {model_path}")
        print(f"Saving vectorizer to: {vectorizer_path}")
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print("Model and vectorizer saved successfully!")
        
        # Verify files were created
        status = check_files_exist()
        print("\nFile status after training:")
        for name, info in status.items():
            print(f"{name}:")
            print(f"  Path: {info['path']}")
            print(f"  Exists: {info['exists']}")
            print(f"  Size: {info['size']} bytes")
        
        # Print intent distribution
        print("\nIntent distribution in the dataset:")
        print(df['intent'].value_counts())
        
        # Print some example preprocessed queries
        print("\nExample of preprocessed queries:")
        print(df[['query', 'processed_query']].head())
        
        # Test the saved model
        print("\nTesting saved model...")
        loaded_model = joblib.load(model_path)
        loaded_vectorizer = joblib.load(vectorizer_path)
        
        test_query = "Hello there"
        test_vec = loaded_vectorizer.transform([preprocess_text(test_query)])
        test_pred = loaded_model.predict(test_vec)[0]
        print(f"Test prediction for '{test_query}': {test_pred}")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise e

if __name__ == "__main__":
    train_model() 