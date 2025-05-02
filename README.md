# Intelligent ML Chatbot

This project implements an intelligent self-learning chatbot using machine learning to provide automated responses to customer queries. The chatbot uses both machine learning classification and fuzzy matching for hybrid prediction, and includes a modern Streamlit user interface.

## Project Structure
```
project_root/
│
├── chatbot_app.py                # Main Streamlit application
├── model_training.py             # ML model training script
├── customer_queries.csv          # Training dataset
├── intent_model.joblib           # Trained ML classifier
├── intent_vectorizer.joblib      # TF-IDF vectorizer for text processing
├── query_corpus.pkl              # Saved query data for fuzzy matching
├── verify_setup.py               # Utility to verify environment setup
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Features
- Hybrid prediction system combining ML classification and fuzzy matching
- Self-learning capability to improve from user feedback
- Modern, responsive Streamlit user interface
- Real-time chat interaction with confidence scores
- User feedback collection and correction mechanism
- Handles misspellings and query variations
- Preprocessing of text for better feature extraction

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify your setup:
   ```bash
   python verify_setup.py
   ```
4. Train the model (or use pre-trained models):
   ```bash
   python model_training.py
   ```
5. Run the chatbot:
   ```bash
   streamlit run chatbot_app.py
   ```

## Requirements
- Python 3.8+
- See requirements.txt for full list of dependencies:
  - streamlit
  - pandas
  - scikit-learn
  - joblib
  - nltk
  - fuzzywuzzy
  - python-Levenshtein
  - matplotlib
  - plotly

## Usage
1. Launch the application using Streamlit
2. Type your query in the chat interface
3. The chatbot will respond based on its training
4. Provide feedback using the thumbs up/down buttons
5. Correct answers will be incorporated for future learning

## Project Highlights
- Intent classification using Logistic Regression with TF-IDF features
- Fuzzy matching for handling misspellings and query variations
- Hybrid prediction system that combines both approaches
- Self-learning mechanism to improve over time
- Comprehensive text preprocessing pipeline
- Modern UI with confidence indicators

## License
MIT License
