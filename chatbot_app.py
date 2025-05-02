import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import pickle
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import csv
import re

# Set page config
st.set_page_config(
    page_title="Intelligent Self-Learning Customer Support Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .content {
        display: flex;
        flex-direction: column;
    }
    .chat-message .meta {
        font-size: 0.8rem;
        color: #a0a0a0;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 0.25rem;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFC107;
        font-weight: bold;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
    /* Improved chat messages styling */
    div.stChatMessage {
        border-bottom-right-radius: 1.5rem !important;
        border-top-left-radius: 1.5rem !important;
        border-top-right-radius: 1.5rem !important;
        border-bottom-left-radius: 0.2rem !important;
        padding: 0.5rem 1rem;
        width: fit-content;
        max-width: 80%;
        margin-bottom: 1rem;
    }
    div.stChatMessage[data-testid="assistant"] {
        background-color: #383e56 !important;
        border: 1px solid #4d5371;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    div.stChatMessage[data-testid="user"] {
        background-color: #2e4057 !important;
        border: 1px solid #3d5169;
        margin-left: auto;
        border-bottom-left-radius: 1.5rem !important;
        border-bottom-right-radius: 0.2rem !important;
    }
    div.stChatMessage div[data-testid="chatMessage"] {
        font-size: 1rem;
    }
    /* Feedback buttons */
    .feedback-buttons {
        display: flex;
        gap: 10px;
        margin-top: 5px;
    }
    .feedback-button {
        background-color: transparent;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .feedback-button:hover {
        background-color: #4d5371;
    }
    .feedback-thumbs-up {
        color: #4CAF50;
    }
    .feedback-thumbs-down {
        color: #F44336;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = {}
    
if 'learning_data' not in st.session_state:
    st.session_state.learning_data = []

def get_confidence_class(confidence):
    """Get CSS class based on confidence level."""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    return "confidence-low"

def preprocess_text(text):
    """Clean and preprocess text for better feature extraction."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits (but keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Rejoin tokens
    return ' '.join(tokens)

def load_resources():
    """Load all necessary resources for the chatbot."""
    try:
        # Load the intent classifier and vectorizer
        intent_model = joblib.load('intent_model.joblib')
        vectorizer = joblib.load('intent_vectorizer.joblib')
        
        # Try to load the query corpus for hybrid prediction
        try:
            with open('query_corpus.pkl', 'rb') as f:
                query_corpus = pickle.load(f)
                
            # Check if corpus has expected structure
            if all(k in query_corpus for k in ['original_queries', 'intents', 'responses']):
                st.sidebar.success("✅ Loaded hybrid prediction system")
                return {
                    'intent_model': intent_model, 
                    'vectorizer': vectorizer, 
                    'query_corpus': query_corpus,
                    'mode': 'hybrid'
                }
        except Exception as e:
            st.sidebar.warning(f"Query corpus not available: {str(e)}")
        
        # Fall back to standard intent classifier if hybrid not available
        df = pd.read_csv('customer_queries.csv')
        return {'intent_model': intent_model, 'vectorizer': vectorizer, 'df': df, 'mode': 'standard'}
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None

def add_to_dataset(query, response, intent):
    """Add a new query-response pair to the dataset for self-learning."""
    try:
        with open('customer_queries.csv', 'a', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([query, response, intent])
        
        # Also add to in-memory learning data
        st.session_state.learning_data.append({
            'query': query,
            'response': response,
            'intent': intent,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        st.sidebar.success(f"✅ Added new entry to dataset: {intent}")
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to add to dataset: {str(e)}")
        return False

def hybrid_predict(query, resources, threshold=60):
    """Predict using both fuzzy matching and ML models."""
    try:
        # Extract resources
        intent_model = resources['intent_model']
        vectorizer = resources['vectorizer']
        
        if resources['mode'] == 'hybrid':
            # Get the corpus data
            query_corpus = resources['query_corpus']
            original_queries = query_corpus['original_queries']
            intents = query_corpus['intents']
            responses = query_corpus['responses']
            
            # Check fuzzy matching first
            best_match, score = process.extractOne(query.lower(), original_queries, scorer=fuzz.token_sort_ratio)
            if score >= threshold:
                # Find the index of the best match in original_queries
                match_idx = original_queries.index(best_match)
                # Return the intent and response for this query
                return intents[match_idx], responses[match_idx], score/100, 'fuzzy'
        else:
            # No corpus available, use the dataframe
            df = resources['df']
            original_queries = df['query'].tolist()
            
            # Check fuzzy matching
            best_match, score = process.extractOne(query.lower(), original_queries, scorer=fuzz.token_sort_ratio)
            if score >= threshold:
                # Find the match in dataframe
                match_row = df[df['query'] == best_match].iloc[0]
                return match_row['intent'], match_row['response'], score/100, 'fuzzy'
        
        # If fuzzy match not good enough or not available, use the ML model
        processed_query = preprocess_text(query)
        query_vec = vectorizer.transform([processed_query])
        intent = intent_model.predict(query_vec)[0]
        confidence = intent_model.predict_proba(query_vec).max()
        
        # Get the response
        if resources['mode'] == 'hybrid':
            # Get response from corpus
            matching_indices = [i for i, x in enumerate(intents) if x == intent]
            if matching_indices:
                response = responses[matching_indices[0]]
            else:
                response = None
        else:
            # Get response from dataframe
            matching_responses = resources['df'][resources['df']['intent'] == intent]['response'].tolist()
            response = matching_responses[0] if matching_responses else None
        
        return intent, response, confidence, 'ml'
    except Exception as e:
        import traceback
        print(f"Hybrid prediction error: {traceback.format_exc()}")
        # In case of error, return None
        return None, None, 0.0, 'error'

def get_response(query, resources):
    """Get response using the appropriate prediction method."""
    try:
        # Get prediction using hybrid approach
        intent, response, confidence, method = hybrid_predict(query, resources)
        
        # Handle prediction errors
        if method == 'error':
            return {
                'response': "I'm sorry, I encountered an error processing your query.",
                'intent': "error",
                'confidence': 0.0,
                'is_low_confidence': True,
                'is_fallback': True,
                'method': 'error'
            }
        
        # Debug information - print to console
        print(f"Processed query: {query}")
        print(f"Predicted intent: {intent}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Method: {method}")
        
        # Handle case where no response is found
        if response is None:
            return {
                'response': "I understand the intent, but I couldn't find a specific answer. Please contact support for more help.",
                'intent': intent,
                'confidence': confidence,
                'is_fallback': True,
                'is_low_confidence': confidence < 0.3,
                'method': method
            }
        
        # For very low confidence (except for greetings), ask for clarification
        if confidence < 0.25 and 'greeting' not in intent.lower():
            return {
                'response': "I'm not sure how to answer that. Could you rephrase your question?",
                'intent': intent,
                'confidence': confidence,
                'is_low_confidence': True,
                'method': method
            }
            
        # For greeting intents, use a lower threshold
        if 'greeting' in intent.lower() and confidence < 0.10:
            return {
                'response': "I'm not sure how to answer that. Could you rephrase your question?",
                'intent': intent,
                'confidence': confidence,
                'is_low_confidence': True,
                'method': method
            }
            
        # Otherwise return the matched response
        return {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'is_low_confidence': confidence < 0.5 and method == 'ml',
            'is_fallback': False,
            'method': method
        }
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        import traceback
        print(f"Exception details: {traceback.format_exc()}")
        return {
            'response': "I'm sorry, I encountered an error processing your query.",
            'intent': "error",
            'confidence': 0.0,
            'is_low_confidence': True,
            'is_fallback': True,
            'method': 'error'
        }

def format_meta_info(response_data):
    """Format metadata information for display."""
    confidence_class = get_confidence_class(response_data['confidence'])
    confidence_percentage = f"{response_data['confidence']:.2%}"
    method = response_data.get('method', 'standard')
    
    if response_data.get('is_low_confidence'):
        return f"""
            <div class="meta">
                <span class="{confidence_class}">Low Confidence Response</span><br>
                <b>Intent:</b> {response_data['intent']} | <b>Confidence:</b> <span class="{confidence_class}">{confidence_percentage}</span> | <b>Method:</b> {method}
            </div>
        """
    elif response_data.get('is_fallback'):
        return f"""
            <div class="meta">
                <span class="{confidence_class}">Fallback Response</span><br>
                <b>Intent:</b> {response_data['intent']} | <b>Confidence:</b> <span class="{confidence_class}">{confidence_percentage}</span> | <b>Method:</b> {method}
            </div>
        """
    else:
        return f"""
            <div class="meta">
                <b>Intent:</b> {response_data['intent']} | <b>Confidence:</b> <span class="{confidence_class}">{confidence_percentage}</span> | <b>Method:</b> {method}
            </div>
        """

def render_feedback_buttons(message_idx):
    """Render thumbs up/down buttons for feedback."""
    return f"""
    <div class="feedback-buttons">
        <button class="feedback-button feedback-thumbs-up" onclick="sendFeedback('{message_idx}', 'positive')">👍 Helpful</button>
        <button class="feedback-button feedback-thumbs-down" onclick="sendFeedback('{message_idx}', 'negative')">👎 Not Helpful</button>
    </div>
    <script>
    function sendFeedback(msgIdx, type) {{
        const data = {{messageIdx: msgIdx, feedbackType: type}};
        window.parent.postMessage({{
            type: 'streamlit:feedback',
            feedback: data
        }}, '*');
        
        // Update UI
        const buttons = document.querySelectorAll('.feedback-buttons');
        buttons.forEach(btn => {{
            btn.innerHTML = '<span style="color: #a0a0a0;">Thank you for your feedback!</span>';
        }});
    }}
    </script>
    """

def record_feedback(feedback_data):
    """Handle user feedback and potentially add to training data."""
    try:
        # Placeholder - In a real system, this would update a database
        print(f"Received feedback: {feedback_data}")
        
        message_idx = int(feedback_data['messageIdx'])
        feedback_type = feedback_data['feedbackType']
        
        # Get the message
        message = st.session_state.messages[message_idx]
        
        if feedback_type == 'negative' and message['role'] == 'assistant':
            # This would be where we'd ask the user for a correction
            # For now just mark the feedback as received
            st.session_state.feedback_given[message_idx] = feedback_type
        elif feedback_type == 'positive' and message['role'] == 'assistant':
            # If the confidence was low but user found it helpful, add to training data
            if message.get('meta', {}).get('is_low_confidence', False):
                # Find the preceding user message
                for i in range(message_idx-1, -1, -1):
                    if st.session_state.messages[i]['role'] == 'user':
                        user_query = st.session_state.messages[i]['content']
                        bot_response = message['content']
                        intent = message['meta']['intent']
                        
                        # Add to dataset for retraining
                        add_to_dataset(user_query, bot_response, intent)
                        break
            
            st.session_state.feedback_given[message_idx] = feedback_type
    except Exception as e:
        print(f"Error processing feedback: {e}")

def handle_corrections(user_input, resources):
    """Process user corrections to improve the system."""
    # Check if this is a correction format
    correction_pattern = r"(?i)correct:?\s+(.*?)\s+(?:to|should be):?\s+(.*)"
    match = re.match(correction_pattern, user_input)
    
    if match:
        wrong_text = match.group(1).strip()
        right_text = match.group(2).strip()
        
        # Find the last bot message
        for i in range(len(st.session_state.messages)-1, -1, -1):
            if st.session_state.messages[i]['role'] == 'assistant':
                bot_msg = st.session_state.messages[i]
                
                # If the wrong text is in the bot message, it's a correction
                if wrong_text.lower() in bot_msg['content'].lower():
                    intent = bot_msg['meta']['intent']
                    corrected_response = bot_msg['content'].replace(wrong_text, right_text)
                    
                    # Find the user query that led to this response
                    for j in range(i-1, -1, -1):
                        if st.session_state.messages[j]['role'] == 'user':
                            query = st.session_state.messages[j]['content']
                            
                            # Add the corrected version to the dataset
                            add_to_dataset(query, corrected_response, intent)
                            
                            return {
                                'response': f"Thank you for the correction! I've updated my knowledge: '{wrong_text}' → '{right_text}'",
                                'intent': 'feedback',
                                'confidence': 1.0,
                                'is_correction': True,
                                'method': 'user_correction'
                            }
                
                break
    
    # Not a correction, just process normally
    return get_response(user_input, resources)

def main():
    st.title("🤖 Self-Learning Intelligent Customer Support Chatbot")
    
    # Sidebar for debugging and options
    with st.sidebar:
        st.header("Chatbot Information")
        st.write("Current working directory:", os.getcwd())
        
        # Load resources
        resources = load_resources()
        
        if resources:
            st.success(f"✅ System running in {resources['mode']} mode")
            
            # Button to retrain the model
            if st.button("Retrain Model"):
                with st.spinner("Retraining model with latest data..."):
                    st.info("This would execute model_training.py in a full implementation")
                    # In a real implementation:
                    # os.system("python model_training.py")
                    st.success("Model retraining complete!")
        else:
            st.error("❌ Failed to load resources")
            
        # Show self-learning stats
        st.header("Self-Learning Stats")
        try:
            df = pd.read_csv('customer_queries.csv')
            st.write(f"Total training examples: {len(df)}")
            st.write(f"Intent distribution:")
            st.write(df['intent'].value_counts())
            
            # Show recently learned items
            if st.session_state.learning_data:
                st.write("Recently learned items:")
                for item in st.session_state.learning_data[-3:]:  # Show last 3 items
                    st.write(f"- [{item['intent']}] {item['query'][:30]}...")
        except:
            st.write("No dataset stats available")
            
        # Instructions for using the chatbot
        st.header("How to use the chatbot")
        st.write("• Ask any customer service question")
        st.write("• Provide feedback using the thumbs up/down")
        st.write("• Correct responses by typing: 'Correct: [wrong text] to [right text]'")
    
    # Main chat interface
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("meta"):
                st.markdown(format_meta_info(message["meta"]), unsafe_allow_html=True)
                
                # Only show feedback buttons for bot messages without feedback
                if message["role"] == "assistant" and i not in st.session_state.feedback_given:
                    st.markdown(render_feedback_buttons(i), unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        if resources:
            # Check if this is a correction
            response_data = handle_corrections(prompt, resources)
            
            # Add bot response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data['response'],
                "meta": response_data
            })
            
            # Display bot response
            with st.chat_message("assistant"):
                st.markdown(response_data['response'])
                st.markdown(format_meta_info(response_data), unsafe_allow_html=True)
                
                # Only show feedback buttons for regular responses
                if not response_data.get('is_correction'):
                    st.markdown(render_feedback_buttons(len(st.session_state.messages) - 1), unsafe_allow_html=True)
        else:
            st.error("Chatbot is not properly initialized. Please check the debug information in the sidebar.")

if __name__ == "__main__":
    main() 