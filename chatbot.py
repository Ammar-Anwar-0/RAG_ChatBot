import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_groq import ChatGroq
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


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
# def load_documents(data_path="data"):
#     docs = []
#     for file in os.listdir(data_path):
#         file_path = os.path.join(data_path, file)
#         if file.endswith(".txt"):
#             loader = TextLoader(file_path)
#         elif file.endswith(".pdf"):
#             loader = PyPDFLoader(file_path)
#         else:
#             continue  # skip unsupported files
#         docs.extend(loader.load())
#     return docs

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

# Retrieval + Answering
# def retrieve_and_answer(query):
#     results = vector_store.similarity_search_with_score(query, k=1)
#     if results and results[0][1] < 0.6:  # threshold for relevance
#         context = results[0][0].page_content
#         prompt = f"""
#         You are a helpful assistant. Use ONLY the following context to answer:
#         {context}
        
#         Question: {query}
#         """
#         return llm.invoke(prompt).content
#     else:
#         return "I can only answer questions related to my knowledge base."

# def retrieve_and_answer(query):
#     results = vector_store.similarity_search_with_score(query, k=5)
    
#     if not results:
#         return "I couldn't find anything relevant in my knowledge base."

#     # Filter based on threshold
#     relevant_results = [(doc, score) for doc, score in results if score >= 0.75]

#     if not relevant_results:
#         return "I'm not confident I can answer that based on the documents I have."

#     # Combine top 3 relevant contexts
#     context = "\n\n".join([doc.page_content for doc, _ in relevant_results[:3]])
    
#     prompt = f"""
# You are a helpful assistant for technical documentation.
# Use ONLY the following context to answer the question.
# If you are unsure or the answer is not in the context, say: "I don't know."

# Context:
# {context}

# Question: {query}
# """
#     return llm.invoke(prompt).content

# def retrieve_and_answer(query):
#     results = vector_store.similarity_search_with_score(query, k=10)

#     if results:
#         # Filter results above similarity threshold
#         relevant_contexts = [
#             doc.page_content
#             for doc, score in results
#             if score <= 0.5
#         ]

#         if relevant_contexts:
#             # Combine top 3 relevant contexts
#             combined_context = "\n\n".join(relevant_contexts[:3])

#             # Log for debugging
#             print("🔍 Similarity Scores:")
#             for i, (doc, score) in enumerate(results[:3]):
#                 print(f"  [{i}] Score: {score:.4f}")
#                 print(f"      Context: {doc.page_content[:200]}...\n")

#             # Construct prompt for the LLM
#             prompt = f"""
# You are a professional assistant, ignore all queries not relevant to our context and try to use the following context to answer the question.
# If you can’t find the answer directly, state it directly and if it is not relevant to context, respond with "I can only answer questions related to my knowledge base."

# Context:
# {combined_context}

# Question: {query}
# """

#             return llm.invoke(prompt).content

#     return "I can only answer questions related to my knowledge base."

def build_prompt(context, query):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a professional assistant. Use ONLY the following context to answer the question.\n"
        "If the answer is not present in the context, say: 'I can only answer questions related to my knowledge base.'\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"### Context:\n{context}\n\n"
        f"### Question:\n{query}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )


def retrieve_and_answer(query):
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



# Streamlit UI
st.title("RAG Chatbot (Groq API)")
query = st.text_input("Ask me something...")

if query:
    st.write(retrieve_and_answer(query))

