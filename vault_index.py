# vault_index.py

import os
import shutil

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: adjust this if your notes live in a different folder
script_dir = os.path.dirname(os.path.abspath(__file__))
vault_path = os.path.join(script_dir, "data")       # Path to your Obsidian markdown files
chroma_folder = os.path.join(script_dir, "chroma_db") # Where we will store ChromaDB’s on-disk files
# ────────────────────────────────────────────────────────────────────────────────

# STEP 0: Remove any old ChromaDB folder so we start fresh each run
if os.path.exists(chroma_folder):
    shutil.rmtree(chroma_folder)

# STEP 1: Load all `.md` files under vault_path (recursively)
loader = DirectoryLoader(vault_path, glob="**/*.md")
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# STEP 2: Split each document into ~500-character chunks (50-char overlap)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# STEP 3: Create a HuggingFace embedding function (for both indexing & retrieval)
# We use the same multilingual model you already know:
hf = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# STEP 4: Instantiate a LangChain Chroma vectorstore, passing in those chunks
# It will under-the-hood create a chromadb.Client and write to `./chroma_db/`.
vectorstore = Chroma.from_documents(
    documents=chunks,          # our list of Document objects (each with .page_content)
    embedding=hf,              # the embedding function
    persist_directory=chroma_folder,
    collection_name="vault",   # name of our collection
)

# STEP 5: Persist the vectorstore to disk (writes DuckDB/Parquet into chroma_db/)
vectorstore.persist()
print(f"\n✅ Vault indexed and stored in ChromaDB at './{chroma_folder}/'")
