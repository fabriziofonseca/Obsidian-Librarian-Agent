# main.py

import os
import numpy as np

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ── STEP 1: Configure your local LLM and Embeddings ────────────────────────────

# 1.a) Ensure Ollama is running in another terminal: `ollama run mistral`
llm = ChatOllama(model="mistral")

# 1.b) Use a multilingual embedding model for better semantic matches.
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ── STEP 2: Load your ChromaDB vector store ────────────────────────────────────

# This folder ("chroma_db") was created earlier by vault_index.py.
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_function,
)
# Increase k from 5 to 10 so we retrieve more candidates
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ── STEP 3: Debug similarity scores for “What are my goals for today?” ─────────

# 3.a) Compute embedding for our exact query
query_text = "What are my goals for today?"
query_emb_list = embedding_function.embed_documents([query_text])
if len(query_emb_list) == 0:
    print("ERROR: failed to compute query embedding.")
    exit(1)
query_emb = np.array(query_emb_list[0])

# 3.b) Fetch all chunk embeddings and documents from ChromaDB
all_data = vectorstore._collection.get()
all_embeddings = all_data["embeddings"]  # list of lists
all_docs = all_data["documents"]          # list of chunk strings
all_ids = all_data["ids"]                 # list of chunk IDs

# 3.c) Define a simple cosine similarity function
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Add a small epsilon to denominator for numerical stability
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)

# 3.d) Compute similarity scores between the query and each chunk
scores = []
for chunk_emb in all_embeddings:
    chunk_vec = np.array(chunk_emb)
    score = cosine_similarity(query_emb, chunk_vec)
    scores.append(score)

# 3.e) Rank top 5 chunks by similarity
ranked = sorted(zip(all_ids, scores, all_docs), key=lambda x: x[1], reverse=True)

print("\nDEBUG: Top 5 similarity scores for “What are my goals for today?”")
for doc_id, score, content in ranked[:5]:
    snippet = content.replace("\n", " ")[:100]
    print(f"  {doc_id:8}  score={score:.4f}  → {snippet!r}")
print("\n(End of DEBUG)\n")

# ── STEP 4: Build a RetrievalQA chain with a “goals = priorities” hint ─────────

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an assistant answering based only on the notes below.
When the user asks about “goals,” treat those as “priorities.”

---------------------
{context}
---------------------

Question: {question}
Answer:""",
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
)

# ── STEP 5: Prompt Loop ─────────────────────────────────────────────────────────

print("🧠 Ask me anything about your notes. Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ["exit", "quit"]:
        break

    # 1) Retrieve top-k chunks
    docs = retriever.get_relevant_documents(question)
    print("\n📄 Retrieved Chunks:")
    if not docs:
        print("  (no chunks found matching your query)\n")
    else:
        for idx, doc in enumerate(docs, start=1):
            snippet = doc.page_content.replace("\n", " ")[:200]
            print(f"[{idx}] {snippet} …\n")

    # 2) Invoke the QA chain with the correct "query" key
    try:
        result = qa_chain.invoke({"query": question})
        answer_text = result["result"]
    except Exception as e:
        answer_text = f"❌ Error running RetrievalQA: {e}"

    # 3) Print the LLM’s response
    print(f"VaultBot: {answer_text}\n")
