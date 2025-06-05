AI Lawyer - Legal Question Answering System


Description:
------------
AI Lawyer is an AI-powered assistant that answers legal questions based on Indian law. It uses Retrieval-Augmented Generation (RAG) to extract relevant legal content from sources like the Constitution of India and the Indian Penal Code.

Technologies Used:
------------------
- LangChain for building the RAG pipeline
- DeepSeek LLM via Ollama for generating responses
- FAISS for vector similarity search
- Streamlit for the user interface
- Sentence Transformers for text embedding

How to Run:
-----------
1. Place your legal PDFs (e.g., Constitution, IPC) in the project folder.
2. Run the vector database script to index documents:
   
   ```bash
   python vector_database.py
