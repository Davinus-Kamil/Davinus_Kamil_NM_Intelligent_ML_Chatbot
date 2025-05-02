# Intelligent ML Chatbot

This project implements an intelligent chatbot using machine learning to provide automated responses to customer queries. The chatbot is built using Python and Streamlit for the user interface.

## Project Structure
```
chatbot_project/
│
├── customer_queries.csv          # Training dataset
├── model_training.py             # ML model training script
├── chatbot_app.py               # Streamlit UI application
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Features
- Machine learning-based response generation
- User-friendly Streamlit interface
- Real-time chat interaction
- Customizable responses based on training data

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python model_training.py
   ```
4. Run the chatbot:
   ```bash
   streamlit run chatbot_app.py
   ```

## Requirements
- Python 3.8+
- See requirements.txt for full list of dependencies

## Usage
1. Launch the application using Streamlit
2. Type your query in the chat interface
3. The chatbot will respond based on its training

## License
MIT License
