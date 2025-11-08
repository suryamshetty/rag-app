# backend/embeddings_store.py

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.config import VECTORSTORE_DIR
from langchain_classic.docstore.document import Document
from typing import List
import os

# Initialize HuggingFace embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_vectorstore(documents: List[Document]):
    """
    Create a FAISS vectorstore from a list of documents and save it locally.
    """
    vector_db = FAISS.from_documents(documents, embeddings_model)
    vector_db.save_local(VECTORSTORE_DIR)
    print(f"✅ Vectorstore saved to {VECTORSTORE_DIR}")
    return vector_db

def load_vectorstore():
    """
    Load a FAISS vectorstore from disk.
    """
    if not os.path.exists(VECTORSTORE_DIR):
        raise ValueError(f"Vectorstore not found at {VECTORSTORE_DIR}. Create it first.")
    
    # Load FAISS locally
    return FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings_model,
        allow_dangerous_deserialization=True  # safe for local files you trust
    )
