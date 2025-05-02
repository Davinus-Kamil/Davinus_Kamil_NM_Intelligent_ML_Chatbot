import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords

def verify_training():
    print("\n=== Model Training Verification ===\n")
    
    # 1. Check if data file exists
    data_file = "customer_queries.csv"
    print(f"1. Checking data file '{data_file}'...")
    if not os.path.exists(data_file):
        print(f"❌ Error: {data_file} not found!")
        return False
    print(f"✓ Found {data_file}")
    
    # 2. Load and verify data
    print("\n2. Loading data...")
    try:
        df = pd.read_csv(data_file)
        print(f"✓ Data loaded successfully")
        print(f"✓ Number of rows: {len(df)}")
        print(f"✓ Columns: {', '.join(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return False
    
    # 3. Train model
    print("\n3. Training model...")
    try:
        # Vectorize text
        vectorizer = TfidfVectorizer(min_df=1)
        X = vectorizer.fit_transform(df['query'])
        y = df['intent']
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        print("✓ Model trained successfully")
        
        # Test prediction
        test_query = "Hello there"
        test_vec = vectorizer.transform([test_query])
        prediction = model.predict(test_vec)[0]
        print(f"✓ Test prediction for '{test_query}': {prediction}")
    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        return False
    
    # 4. Save model and vectorizer
    print("\n4. Saving model and vectorizer...")
    try:
        joblib.dump(model, 'chatbot_model.joblib')
        joblib.dump(vectorizer, 'vectorizer.joblib')
        print("✓ Files saved successfully")
    except Exception as e:
        print(f"❌ Error saving files: {str(e)}")
        return False
    
    # 5. Verify saved files
    print("\n5. Verifying saved files...")
    model_path = 'chatbot_model.joblib'
    vectorizer_path = 'vectorizer.joblib'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found after saving!")
        return False
    if not os.path.exists(vectorizer_path):
        print(f"❌ Error: {vectorizer_path} not found after saving!")
        return False
        
    print(f"✓ Model file exists: {model_path} ({os.path.getsize(model_path)} bytes)")
    print(f"✓ Vectorizer file exists: {vectorizer_path} ({os.path.getsize(vectorizer_path)} bytes)")
    
    # 6. Test loading saved files
    print("\n6. Testing saved model...")
    try:
        loaded_model = joblib.load(model_path)
        loaded_vectorizer = joblib.load(vectorizer_path)
        
        # Test prediction with loaded model
        test_query = "Hello there"
        test_vec = loaded_vectorizer.transform([test_query])
        prediction = loaded_model.predict(test_vec)[0]
        print(f"✓ Test prediction with loaded model for '{test_query}': {prediction}")
    except Exception as e:
        print(f"❌ Error testing saved model: {str(e)}")
        return False
    
    print("\n✅ All verifications passed successfully!")
    return True

if __name__ == "__main__":
    verify_training() 