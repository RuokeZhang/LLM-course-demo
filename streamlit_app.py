import streamlit as st
from chatbot import Chatbot

# Page config
st.set_page_config(page_title="Multi-Agent Chatbot", page_icon="🤖")

st.title("🤖 Multi-Agent RAG Chatbot Demo")
st.markdown("This chatbot uses Pinecone + OpenAI + multi-agent control logic.")

# Initialize chatbot once
if "bot" not in st.session_state:
    st.session_state.bot = Chatbot()

# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
user_input = st.chat_input("Ask something about the machine-learning document...")

if user_input:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    response = st.session_state.bot.chat(user_input)

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)