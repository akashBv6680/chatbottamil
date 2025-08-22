import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set the background color
st.markdown("""
<style>
.stApp {
  background-color: #f0f2f5; 
}
</style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_translator():
    checkpoint = "Mr-Vicky-01/English-Tamil-Translator"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return tokenizer, model

tokenizer, model = load_translator()

def language_translator(text):
    tokenized = tokenizer([text], return_tensors='pt')
    out = model.generate(**tokenized, max_length=128)
    return tokenizer.decode(out[0], skip_special_tokens=True)

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
