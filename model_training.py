import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pickle

# Download necessary NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("NLTK download failed but continuing anyway")

def preprocess_text(text):
    """Clean and preprocess text for better feature extraction."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits (but keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
    except:
        pass  # If stopwords fail, continue without them
    
    # Stemming
    try:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    except:
        pass  # If stemming fails, continue without it
    
    # Rejoin tokens
    return ' '.join(tokens)

def train_intent_classifier():
    # Load the data
    df = pd.read_csv('customer_queries.csv')
    
    # Fix the 'greeting ' intent (with extra space) to 'greeting'
    df['intent'] = df['intent'].str.strip()
    
    # Create a preprocessed query column
    df['processed_query'] = df['query'].apply(preprocess_text)
    
    # Print dataset statistics
    print(f"Loaded dataset with {len(df)} records")
    print(f"Number of unique intents: {df['intent'].nunique()}")
    print("Intent distribution:")
    intent_counts = df['intent'].value_counts()
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count} examples")
    
    # Create TF-IDF vectorizer with improved parameters
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=1,
        max_df=0.9,
        stop_words='english',
        ngram_range=(1, 3),
        norm='l2',
        sublinear_tf=True
    )
    
    # Transform the preprocessed queries
    X = vectorizer.fit_transform(df['processed_query'])
    y = df['intent']
    
    # Print vectorizer info
    print(f"\nFeature matrix shape: {X.shape}")
    
    # Due to small sample size, we'll train on all data rather than splitting
    # This is acceptable for a prototype/demo system
    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver='liblinear',
        class_weight='balanced',
        multi_class='ovr'
    )
    model.fit(X, y)
    
    # Instead of train/test split, evaluate with 3-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=3)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
    
    # Save everything needed for prediction
    joblib.dump(model, 'intent_model.joblib')
    joblib.dump(vectorizer, 'intent_vectorizer.joblib')
    
    # Save the query corpus for fuzzy matching
    query_data = {
        'original_queries': df['query'].tolist(),
        'intents': df['intent'].tolist(),
        'responses': df['response'].tolist()
    }
    with open('query_corpus.pkl', 'wb') as f:
        pickle.dump(query_data, f)
    
    print("\nModel, vectorizer, and query corpus saved successfully!")
    
    # Test the model with some examples including misspellings
    test_queries = [
        "What time do you open?",
        "I need to know where my package is",
        "Can I get my money back?",
        "How do I change my password?",
        "Hi there",
        "Hey",
        "What is your retrun policy?",  # Misspelled
        "Payment metods?",  # Misspelled
        "Shpping to Europe?",  # Misspelled
        "Waranty information"  # Misspelled
    ]
    
    # Load the corpus data for testing
    original_queries = df['query'].tolist()
    
    print("\nTesting with example queries (including misspellings):")
    for query in test_queries:
        # First test fuzzy matching
        best_match, score = process.extractOne(query.lower(), original_queries, scorer=fuzz.token_sort_ratio)
        print(f"\nQuery: {query}")
        print(f"Best fuzzy match: {best_match} (score: {score})")
        
        # Then test the model
        processed_query = preprocess_text(query)
        query_vec = vectorizer.transform([processed_query])
        intent = model.predict(query_vec)[0]
        proba = model.predict_proba(query_vec)[0]
        confidence = proba.max()
        
        # Find top 3 intents with probabilities
        top_indices = proba.argsort()[-3:][::-1]
        top_intents = [(model.classes_[i], proba[i]) for i in top_indices]
        
        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence:.4f}")
        print("Top 3 intents:")
        for i, (intent_name, score) in enumerate(top_intents, 1):
            print(f"  {i}. {intent_name}: {score:.4f}")
    
    # Test hybrid prediction
    print("\nTesting hybrid prediction:")
    for query in test_queries:
        # Using hybrid approach directly instead of through a function
        best_match, score = process.extractOne(query.lower(), original_queries, scorer=fuzz.token_sort_ratio)
        
        if score >= 60:  # Threshold for fuzzy matching
            match_idx = original_queries.index(best_match)
            intent = df['intent'].iloc[match_idx]
            response = df['response'].iloc[match_idx]
            confidence = score/100
        else:
            processed_query = preprocess_text(query)
            query_vec = vectorizer.transform([processed_query])
            intent = model.predict(query_vec)[0]
            confidence = model.predict_proba(query_vec).max()
            
            # Get the response for this intent
            responses = df[df['intent'] == intent]['response'].tolist()
            response = responses[0] if responses else None
            
        print(f"\nQuery: {query}")
        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Response: {response[:50]}..." if response else "No response found")

if __name__ == "__main__":
    train_intent_classifier() 