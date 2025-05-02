print("Testing imports...")
try:
    import streamlit
    print("✓ Streamlit imported successfully")
except Exception as e:
    print(f"✗ Error importing streamlit: {str(e)}")

try:
    import joblib
    print("✓ Joblib imported successfully")
except Exception as e:
    print(f"✗ Error importing joblib: {str(e)}")

try:
    import sklearn
    print("✓ Scikit-learn imported successfully")
except Exception as e:
    print(f"✗ Error importing sklearn: {str(e)}")

try:
    import nltk
    print("✓ NLTK imported successfully")
except Exception as e:
    print(f"✗ Error importing nltk: {str(e)}")

print("\nChecking if model files exist...")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'chatbot_model.joblib')
vectorizer_path = os.path.join(script_dir, 'vectorizer.joblib')

print(f"Current directory: {script_dir}")
print(f"Model path exists: {os.path.exists(model_path)}")
print(f"Vectorizer path exists: {os.path.exists(vectorizer_path)}")

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    print("\nTrying to load model and vectorizer...")
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("✓ Model and vectorizer loaded successfully!")
        
        # Test a prediction
        test_query = "Hello there"
        from model_training import preprocess_text
        cleaned = preprocess_text(test_query)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        print(f"\nTest prediction:")
        print(f"Query: {test_query}")
        print(f"Predicted intent: {pred}")
    except Exception as e:
        print(f"✗ Error loading model/vectorizer: {str(e)}")
else:
    print("\n✗ Model files not found. Need to train the model first.") 