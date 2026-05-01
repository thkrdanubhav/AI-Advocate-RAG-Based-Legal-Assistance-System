from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embedding_model import get_embedding_model

# List your legal PDFs here
pdf_files = ["the_constitution_of_india.pdf", "indian_penal_code.pdf"]

print("Loading legal documents and splitting into chunks...")

documents = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(documents)

embedding_model = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
faiss_db = FAISS.from_documents(text_chunks, embedding_model)


faiss_db.save_local("faiss_index")

print(" FAISS index created and saved successfully with multiple legal sources.")



