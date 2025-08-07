# AWS Service Catalog RAG Chatbot
#### This is an intelligent chatbot that leverages Retrieval-Augmented Generation (RAG) to assist users with questions related to the AWS Service Catalog Developer Guide. It uses:
- LangChain for orchestration
- FAISS as the vector store
- HuggingFace Transformers for embeddings and reranking
- Groq API to generate fast, high-quality responses
- Streamlit for a responsive and interactive user interface

## Features
- Semantic search using Vector Embeddings
- Reranker using 'cross-encoder/ms-marco-MiniLM-L-6-v2' for more accurate results
- User must press button before response and history of conversation will be displayed.
- Use of Llama3-70b-8192.

## Project Structure
|- chatbot.py                          # Rag Logic + Prompting + SteamlitUI

|- data|- amazon-service-datalog-dg.pdf   # Your PDF / Text File

|- .env                                # store your GROQ_API_Key here (WARNING: Never make this public)

|- requirements.txt                    # All your dependencies (run command: pip freeze > requirements.txt
#### You can optimize the structure by adding a main.py file for Streamlit UI.
