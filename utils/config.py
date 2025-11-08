# utils/config.py
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MODEL_NAME = "gpt-4o-mini"   # change to model you have access to, e.g. "gpt-4-turbo"
