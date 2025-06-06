# Obsidian Librarian-Agent

This project indexes the Markdown notes under the `data/` folder and lets you
query them using a local LLM. Paths to the notes and ChromaDB index are resolved
relative to each script, so you can run the tools from any working directory.

## Usage

1. **Index the vault**

   ```bash
   python vault_index.py
   ```

   This reads all `*.md` files in `data/` and stores embeddings in
   `chroma_db/`.

2. **Start chatting**

   ```bash
   python main.py
   ```

   Ask questions about your notes. Type `exit` to quit.

