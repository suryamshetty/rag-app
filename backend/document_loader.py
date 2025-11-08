from pathlib import Path
from langchain_classic.docstore.document import Document
from typing import List

def load_txts_from_folder(folder_path: str) -> List[Document]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder path {folder_path} does not exist or is not a directory.")
    
    documents = []
    for file_path in folder.glob("*.txt"):  # automatically finds all .txt files
        text = file_path.read_text(encoding="utf-8").strip()
        if text:  # skip empty files
            documents.append(Document(page_content=text, metadata={"source": file_path.name}))
    
    return documents
