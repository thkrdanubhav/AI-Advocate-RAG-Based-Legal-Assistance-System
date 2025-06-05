import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-r1:1.5b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_NAME = os.getenv("INDEX_NAME", "faiss_index")

# STEP 1: Create FAISS index
def create_vector_store(pdf_path, index_path=INDEX_NAME):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(index_path)
    print("✅ Vector index created and saved.")

# STEP 2: Respond to questions
def get_response(query):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(INDEX_NAME, embeddings=embeddings, allow_dangerous_deserialization=True)

    llm = Ollama(model=MODEL_NAME)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    result = qa_chain.invoke({"query": query})
    return result["result"]



