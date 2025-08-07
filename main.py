import streamlit as st
from chatbot import retrieve_and_answer  

st.set_page_config(page_title="AWS Service Catalog Assistant", page_icon=":robot_face:")

# Initialize in-memory chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("AWS Service Catalog Developer Guide Assistant")
st.write("Ask me anything about the AWS Service Catalog Developer Guide. "
         "I will only answer questions based on the uploaded documentation.")

# User input
query = st.text_input("Your question:")

# Submit button
if st.button("Submit Question"):
    if query.strip():
        response = retrieve_and_answer(query)
        st.session_state.chat_history.append({"user": query, "bot": response})
    else:
        st.warning("Please enter a question before submitting.")

# Display chat history
if st.session_state.chat_history:
    st.subheader("Conversation History")
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Assistant:** {chat['bot']}")
        st.write("---")

# Clear chat history
if st.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()
