import os
from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import sys

def verify_environment():
    """Verify the Python environment and dependencies."""
    print("\n=== Environment Verification ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).absolute()}")
    
    # Check required packages
    required_packages = ['streamlit', 'joblib', 'pandas', 'scikit-learn', 'nltk']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")

def verify_files():
    """Verify required files exist and are accessible."""
    print("\n=== File Verification ===")
    base_dir = Path(__file__).parent.absolute()
    
    required_files = [
        ('Dataset', 'customer_queries.csv'),
        ('Model', 'chatbot_model.joblib'),
        ('App', 'chatbot_app.py'),
        ('Training Script', 'model_training.py')
    ]
    
    for name, filename in required_files:
        path = base_dir / filename
        if path.exists():
            print(f"✓ {name} found at {path}")
            print(f"  Size: {path.stat().st_size / 1024:.2f} KB")
            print(f"  Last modified: {path.stat().st_mtime}")
        else:
            print(f"✗ {name} NOT found at {path}")

def verify_model():
    """Verify model can be loaded and used."""
    print("\n=== Model Verification ===")
    model_path = Path(__file__).parent / 'chatbot_model.joblib'
    
    try:
        model = joblib.load(model_path)
        print("✓ Model loaded successfully")
        
        # Test prediction
        test_query = "Hello, how are you?"
        from model_training import preprocess_text
        processed_query = preprocess_text(test_query)
        prediction = model.predict([processed_query])[0]
        print(f"✓ Test prediction successful")
        print(f"  Input: '{test_query}'")
        print(f"  Output: '{prediction}'")
        
    except Exception as e:
        print(f"✗ Error loading/testing model: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    print("Starting verification process...")
    verify_environment()
    verify_files()
    verify_model()
    print("\nVerification complete!") 