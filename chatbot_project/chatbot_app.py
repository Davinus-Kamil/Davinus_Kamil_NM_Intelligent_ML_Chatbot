import streamlit as st
import joblib
import string
import os
import nltk
from model_training import preprocess_text

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'chatbot_model.joblib')
VECTORIZER_PATH = os.path.join(SCRIPT_DIR, 'vectorizer.joblib')

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Set page config
st.set_page_config(
    page_title="Intelligent Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Debug Information in Sidebar
st.sidebar.header("🔍 Debug Information")

# Directory Information
st.sidebar.subheader("Directory Information")
st.sidebar.code(f"Script Directory:\n{SCRIPT_DIR}")
st.sidebar.code(f"Current Working Directory:\n{os.getcwd()}")

# File Paths
st.sidebar.subheader("File Paths")
st.sidebar.code(f"Model Path:\n{MODEL_PATH}")
st.sidebar.code(f"Vectorizer Path:\n{VECTORIZER_PATH}")

# File Status
st.sidebar.subheader("File Status")
model_exists = os.path.exists(MODEL_PATH)
vectorizer_exists = os.path.exists(VECTORIZER_PATH)

st.sidebar.markdown(f"✓ Model File Exists: `{model_exists}`")
if model_exists:
    st.sidebar.markdown(f"✓ Model Size: `{os.path.getsize(MODEL_PATH)}` bytes")
else:
    st.sidebar.error("⚠️ Model file not found!")

st.sidebar.markdown(f"✓ Vectorizer File Exists: `{vectorizer_exists}`")
if vectorizer_exists:
    st.sidebar.markdown(f"✓ Vectorizer Size: `{os.path.getsize(VECTORIZER_PATH)}` bytes")
else:
    st.sidebar.error("⚠️ Vectorizer file not found!")

# Custom CSS
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

def load_model_and_vectorizer():
    """Load the trained model and vectorizer."""
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(f"Vectorizer file not found at: {VECTORIZER_PATH}")

        # Load model with detailed error handling
        try:
            model = joblib.load(MODEL_PATH)
            st.sidebar.success("✓ Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error loading model: {str(e)}")
            return None, None

        # Load vectorizer with detailed error handling
        try:
            vectorizer = joblib.load(VECTORIZER_PATH)
            st.sidebar.success("✓ Vectorizer loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error loading vectorizer: {str(e)}")
            return None, None

        return model, vectorizer

    except Exception as e:
        st.error(f"⚠️ Error in load_model_and_vectorizer: {str(e)}")
        st.error("Please ensure you have trained the model by running:")
        st.code("python model_training.py")
        return None, None

def main():
    st.title("🤖 Intelligent Chatbot")
    st.markdown("Welcome! I'm your AI assistant. How can I help you today?")

    # Load the model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    if model is None or vectorizer is None:
        st.warning("Please train the model first by running the training script.")
        st.code("python model_training.py")
        return

    # Input query
    query = st.text_input("Ask me something:")

    if query:
        try:
            # Preprocess the query
            cleaned = preprocess_text(query)
            
            # Vectorize and predict
            vec = vectorizer.transform([cleaned])
            intent = model.predict(vec)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            # Create a nice prediction box
            st.markdown(f"""
                <div class="prediction-box">
                    <h4>Query:</h4>
                    <p>{query}</p>
                    <h4>Cleaned Text:</h4>
                    <p>{cleaned}</p>
                    <h4>Predicted Intent:</h4>
                    <p><strong>{intent}</strong></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Add some context about the intent
            if intent == 'greeting':
                st.info("Hello! How can I assist you today?")
            elif intent == 'faq':
                st.info("Let me help you with that question. Here's what I found in our FAQ database.")
            elif intent == 'ticket':
                st.info("I understand you need support. I'll help create a ticket for our support team to assist you.")
                
        except Exception as e:
            st.error(f"⚠️ Error processing query: {str(e)}")
            st.error("Stack trace:", exc_info=True)

if __name__ == "__main__":
    main() 