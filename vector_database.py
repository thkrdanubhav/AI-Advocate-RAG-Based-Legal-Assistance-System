from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embedding_model import get_embedding_model

# ✅ List your legal PDFs here
pdf_files = ["the_constitution_of_india.pdf", "indian_penal_code.pdf"]

print("🔄 Loading legal documents and splitting into chunks...")

# Step 1: Load all PDFs
documents = []
for pdf_path in pdf_files:
    loader = UnstructuredPDFLoader(pdf_path)
    documents.extend(loader.load())

# Step 2: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(documents)

# Step 3: Embed and store in FAISS
embedding_model = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
faiss_db = FAISS.from_documents(text_chunks, embedding_model)

# Step 4: Save FAISS index locally
faiss_db.save_local("faiss_index")

print("✅ FAISS index created and saved successfully with multiple legal sources.")



