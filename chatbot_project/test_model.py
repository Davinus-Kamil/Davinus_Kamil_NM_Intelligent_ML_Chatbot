import os
import joblib
from model_training import preprocess_text

def test_model():
    print("Starting model test...")
    
    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Try to load the model and vectorizer
        print("\nAttempting to load model and vectorizer...")
        print(f"Looking in directory: {script_dir}")
        
        model_path = os.path.join(script_dir, 'chatbot_model.joblib')
        vectorizer_path = os.path.join(script_dir, 'vectorizer.joblib')
        
        print(f"Model path: {model_path}")
        print(f"Vectorizer path: {vectorizer_path}")
        
        # Check if files exist
        print(f"\nChecking if files exist:")
        print(f"Model file exists: {os.path.exists(model_path)}")
        print(f"Vectorizer file exists: {os.path.exists(vectorizer_path)}")
        
        # Load the files
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        print("\nModel and vectorizer loaded successfully!")
        
        # Test some predictions
        test_queries = [
            "Hello there",
            "How do I reset my password?",
            "I need help with my order"
        ]
        
        print("\nTesting predictions:")
        for query in test_queries:
            # Preprocess
            cleaned = preprocess_text(query)
            # Vectorize
            vec = vectorizer.transform([cleaned])
            # Predict
            intent = model.predict(vec)[0]
            print(f"\nQuery: {query}")
            print(f"Cleaned: {cleaned}")
            print(f"Predicted Intent: {intent}")
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    test_model() 