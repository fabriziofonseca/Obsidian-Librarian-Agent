# vault_index_and_chat.py

import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import chromadb
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

script_dir = os.path.dirname(os.path.abspath(__file__))

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: adjust this if your notes live in a different folder
vault_path = os.path.join(script_dir, "data")  # Path to your Obsidian markdown notes
# ────────────────────────────────────────────────────────────────────────────────

# STEP 1: Load every Markdown file under vault_path
loader = DirectoryLoader(vault_path, glob="**/*.md")
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# STEP 2: Split each document into ~500‐character chunks (50‐character overlap)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# STEP 3: Choose a multilingual embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# STEP 4: Build an in‐memory ChromaDB client (no persistence)
#         Everything lives in RAM and will be discarded when the script exits.
client = chromadb.Client()  # default, in‐memory only

# If “vault” exists from a previous run, delete it
existing_collections = [c.name for c in client.list_collections()]
if "vault" in existing_collections:
    client.delete_collection("vault")

# Create a fresh “vault” collection
collection = client.create_collection(name="vault")

# STEP 5: Embed each chunk and add it to ChromaDB (in memory)
for i, chunk in enumerate(chunks):
    text = chunk.page_content
    # Optional: print some of the chunk so you see “Top 3 Priorities”
    snippet = text.replace("\n", " ")[:200]
    print(f"[INDEXING CHUNK {i+1}] {snippet!r}")
    embedding = model.encode(text).tolist()
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[f"doc-{i}"]
    )

# STEP 6: Wrap this in‐memory index for LangChain
# We need an embedding function for the retriever to normalize vector shapes,
# even though the vectors are already in memory.
retriever_embedding_fn = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = chromadb.utils.chroma_client_to_vectorstore(
    client=client,
    collection_name="vault",
    embedding_function=retriever_embedding_fn
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# STEP 7: Start your local Ollama LLM (make sure you ran `ollama run mistral` already)
llm = ChatOllama(model="mistral")

# STEP 8: Build a RetrievalQA chain (using “stuff” to pack top‐k chunks into the prompt)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff"
)

# STEP 9: Interactive chat loop
print("\n🧠 Indexing complete. You can now ask questions (type 'exit' to quit).\n")
while True:
    question = input("You: ").strip()
    if question.lower() in ["exit", "quit"]:
        break

    # 1) Fetch top‐k chunks
    docs = retriever.get_relevant_documents(question)
    print("\n📄 Retrieved Chunks:")
    if not docs:
        print("  (no matching chunks found)\n")
    else:
        for idx, doc in enumerate(docs, start=1):
            snippet = doc.page_content.replace("\n", " ")[:200]
            print(f"[{idx}] {snippet} …\n")

    # 2) Ask the RetrievalQA chain to answer strictly based on those chunks
    try:
        result = qa_chain.invoke({"query": question})
        print(f"VaultBot: {result['result']}\n")
    except Exception as e:
        print(f"VaultBot: ❌ Error running QA chain: {e}\n")
