# src/utils/chroma_setup.py

import os
import json
from dotenv import load_dotenv

import openai
import chromadb

def main():
    # Load OpenAI API key from environment
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    chromadb_dir= os.getenv("CHROMA_DIR")


    if not openai.api_key:
        raise ValueError("Please set OPENAI_API_KEY in your .env file")

    # Initialize ChromaDB (persist files under data/embeddings/)
    client_chroma = chromadb.PersistentClient(path=chromadb_dir)

    # Create or open the 'book_summaries' collection
    collection = client_chroma.get_or_create_collection(name="book_summaries")

    # Load book summaries from JSON file
    with open("../../../data/book_summaries.json", "r", encoding="utf-8") as f:
        books = json.load(f)

    ids = []
    metadata = []
    documents = []
    embeddings = []

    for book in books:
        title = book["title"]
        summary = book["summary"]
        themes = book.get("themes", [])

        # Prepare text for embedding: summary plus themes
        text_for_embedding = summary + "\n\nThemes: " + ", ".join(themes)

        # Call OpenAI Embeddings API
        response = openai.embeddings.create(
            input=[text_for_embedding],
            model="text-embedding-3-small"
        )

        vector = response.data[0].embedding

        # Collect data for insertion into ChromaDB
        ids.append(title)
        metadata.append({
            "title": title,
            # Chroma metadata must be scalar — store a string (or json.dumps(themes))
            "themes": ", ".join(themes),
            "theme_count": len(themes),  # optional scalar
        })
        documents.append(summary)
        embeddings.append(vector)

    # Add all documents to the collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadata,
        documents=documents
    )

    print(f"✅ Successfully loaded {len(ids)} items into ChromaDB at data/embeddings/")

if __name__ == "__main__":
    main()
