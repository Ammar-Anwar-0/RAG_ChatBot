import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to load documents from a directory
def load_documents(data_path="data"):
    docs = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if file.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            continue  # skip unsupported files
        loaded = loader.load()
        print(f"Loaded {len(loaded)} docs from {file}")
        docs.extend(loaded)
    return docs

# Prepare vector store
def create_vector_store():
    raw_docs = load_documents()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(raw_docs)
    return Chroma.from_documents(split_docs, embeddings)

# Initialize
vector_store = create_vector_store()
llm = ChatGroq(model_name="llama3-70b-8192", api_key=GROQ_API_KEY)

# Function to retrieve relevant documents and answer the query
def retrieve_and_answer(query):
    results = vector_store.similarity_search_with_score(query, k=10)

    if results:
        # Filter results above similarity threshold
        relevant_contexts = [
            doc.page_content
            for doc, score in results
            if score >= 0.65
        ]

        if relevant_contexts:
            # Combine top 3 relevant contexts
            combined_context = "\n\n".join(relevant_contexts[:3])

            # Log for debugging
            print("🔍 Similarity Scores:")
            for i, (doc, score) in enumerate(results[:2]): # Limit to first 2 for reference
                print(f"  [{i}] Score: {score:.4f}")
                print(f"      Context: {doc.page_content[:200]}...\n")

            # Construct prompt for the LLM
            prompt = f"""
You are a helpful assistant. Try to use the following context to answer the question.
If you can’t find the answer directly, make an educated guess or explain what’s likely true.

Context:
{combined_context}

Question: {query}
"""

            return llm.invoke(prompt).content

    return "I can only answer questions related to my knowledge base."

# Streamlit UI
st.title("RAG Chatbot (Groq API)")
query = st.text_input("Ask me something...")

if query:
    st.write(retrieve_and_answer(query))
