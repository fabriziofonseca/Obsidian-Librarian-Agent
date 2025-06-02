**Status::** 🟡 In Progress  
**Tags::** #project  #coding #in-progress 
**Start Date::** 2025-05-30  
**Due Date::**  
**Category::** 

---

## 🗂️ Information:
The structure of the project will be this one: 

Load the notes as a .md from the Obsidian Vault path. 
  ```python
loader = DirectoryLoader(vault_path, glob="**/*.md")
documents = loader.load()
```

Breakdown into pieces each note, so it can be handled for the language model. This will breakdown every chunk into ~500 characters with a 50-char overlap so information is not lost on the between pieces. 
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
```

Each chunk is converted into a vector using an embedding model. A vector is a list of numbers that represents meaning. In this case the meaning of the text. 
  ```python
embedding = model.encode(text).tolist()
```

Sentences that mean similar will have similar vectors. Then the AI will be able to connect semantically related information using related vectors. For this purpose we are using `all-MiniLM-L6-v2`

These vector will be stored in ChromeDB that runs locally, this will allow the AI to search by meaning later. 
```python
collection.add(documents=[text], embeddings=[embedding], ids=[f"doc-{i}"])
```



---

## ✅ Tasks
- [ ] 
- [ ] 
- [ ] 