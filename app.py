# app.py
import streamlit as st
import os
from backend.document_loader import load_txts_from_folder  # New loader for txt
from backend.text_splitter import split_documents
from backend.embeddings_store import create_vectorstore, load_vectorstore
from backend.retriever_chain import build_qa_chain
from utils.config import DATA_DIR, VECTORSTORE_DIR
import langchain
# Temporary compatibility patch for old verbose flag
if not hasattr(langchain, "verbose"):
    langchain.verbose = False
if not hasattr(langchain, "debug"):
    langchain.debug = False
if not hasattr(langchain, "llm_cache"):
    langchain.llm_cache = None


st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot (Streamlit)")

st.sidebar.header("Upload & Build")
uploaded_files = st.sidebar.file_uploader("Upload TXT files", accept_multiple_files=True, type=["txt"])
process = st.sidebar.button("Process uploads")

if process and uploaded_files:
    os.makedirs(DATA_DIR, exist_ok=True)
    saved_paths = []
    with st.spinner("Saving and loading TXT files..."):
        for f in uploaded_files:
            tmp_path = os.path.join(DATA_DIR, f.name)
            with open(tmp_path, "w", encoding="utf-8") as out:
                out.write(f.read().decode("utf-8"))
            saved_paths.append(tmp_path)

    st.sidebar.success(f"Saved {len(saved_paths)} files to {DATA_DIR}")

    # Load + split
    with st.spinner("Loading documents..."):
        docs = load_txts_from_folder(DATA_DIR)
    with st.spinner("Splitting into chunks..."):
        chunks = split_documents(docs)

    # Create embeddings & vectorstore
    with st.spinner("Creating embeddings and vector store..."):
        create_vectorstore(chunks)
    st.success("Vectorstore created! Ready to ask questions.")

# Chat interface
if os.path.isdir(VECTORSTORE_DIR):
    st.subheader("Ask a question")
    query = st.text_input("Question:")
    if st.button("Get Answer") and query.strip():
        with st.spinner("Retrieving and generating answer..."):
            vector_db = load_vectorstore()
            qa = build_qa_chain(vector_db)
            result = qa.query(query)       # <-- use .query() instead of .run()
            answer = result["answer"]      # <-- extract the answer text
        st.markdown("**Answer:**")
        st.write(answer)
else:
    st.info("Upload TXT files and click 'Process uploads' in the sidebar to prepare the vector store.")