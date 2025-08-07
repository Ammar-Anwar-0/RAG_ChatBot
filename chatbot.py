import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_groq import ChatGroq
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# For History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load reranker model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Reranking function
def rerank(query, documents, top_k=3):
    pairs = [(query, doc.page_content) for doc in documents]
    
    # Tokenize and score
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)
    
    # Sort by descending score
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents from 'data/' folder
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
    return FAISS.from_documents(split_docs, embeddings)

# Initialize 
def initialize_vector_store():
    return create_vector_store()

def get_llm():
    return ChatGroq(model_name="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

# Prompt (Structured Llama 3)
def build_prompt(context, query):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a professional assistant developed to help users with content from the AWS Service Catalog Developer Guide.\n"
        "You work specifically within the AWS ecosystem and should answer ONLY questions related to AWS Service Catalog.\n"
        "Use ONLY the context provided to answer user questions accurately and concisely while responding like a human.\n"
        "If the context does not contain the answer, respond with: 'I can only answer questions related to my knowledge base.'\n"
        "Politely decline any queries unrelated to AWS Service Catalog or outside your knowledge base.\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"### Context:\n{context}\n\n"
        f"### Question:\n{query}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )

# Retrieval + Answering
def retrieve_and_answer(query):

    vector_store = initialize_vector_store()
    llm = get_llm()

    # Step 1: Retrieve top-k candidates using vector similarity
    results = vector_store.similarity_search_with_score(query, k=10)

    if results:
        # Step 2: Extract the document objects 
        retrieved_docs = [doc for doc, _ in results]

        # Step 3: Rerank using a cross-encoder
        top_docs = rerank(query, retrieved_docs, top_k=3)

        # Step 4: Combine top reranked documents into a single context
        combined_context = "\n\n".join([doc.page_content for doc in top_docs])

        # Optional: Print debug info
        print("📊 Reranked Top 3 Contexts:")
        for i, doc in enumerate(top_docs):
            print(f"[{i}] {doc.page_content[:200]}...\n")

        # Step 5: Formulate prompt
        prompt = build_prompt(combined_context, query)
        return llm.invoke(prompt).content

    return "I can only answer questions related to my knowledge base."

