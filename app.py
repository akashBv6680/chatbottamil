import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from langsmith import traceable, Client

# Initialize the LangSmith client. This will allow us to create runs.
# The API key and project name should be set as environment variables
# or in Streamlit secrets.
client = Client()

# Set the background color
st.markdown("""
<style>
.stApp {
    background-color: #fbf8f0;
}
</style>
""", unsafe_allow_html=True)

# Load model and tokenizer
# The @st.cache_resource decorator caches the model to prevent reloading
# on every rerun, which is essential for performance.
@st.cache_resource
def load_translator():
    checkpoint = "Mr-Vicky-01/English-Tamil-Translator"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return tokenizer, model

# Decorate the translation function with @traceable from LangSmith.
# This will automatically log the function's inputs, outputs, and metadata
# to your LangSmith project, making it easy to debug and monitor.
@traceable(run_type="llm")
def language_translator(text: str):
    """
    Translates English text to Tamil using a pre-trained model.
    The function is decorated to be traced by LangSmith.
    """
    try:
        tokenized = tokenizer([text], return_tensors='pt')
        out = model.generate(**tokenized, max_length=128)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        # Log any errors to the trace
        st.error(f"Translation failed: {e}")
        return f"Error: Translation failed with error '{e}'."

# Load the model and tokenizer only once
tokenizer, model = load_translator()

# Chatbot interface
st.title("English to Tamil Translator Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter a message to translate..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display bot response (translation)
    with st.chat_message("assistant"):
        with st.spinner("Translating..."):
            translated_text = language_translator(prompt)
            st.markdown(f"**Tamil Translation:** {translated_text}")
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": f"**Tamil Translation:** {translated_text}"})
