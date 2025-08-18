import chromadb

def get_collection():
    """
    Get the ChromaDB collection for book summaries.
    If it doesn't exist, create it.
    """
    # Initialize ChromaDB client (persist files under data/embeddings/)
    client_chroma = chromadb.PersistentClient(path="../../../data/embeddings")

    # Create or open the 'book_summaries' collection
    collection = client_chroma.get_or_create_collection(name="book_summaries")

    results = collection.get()

    return results

if __name__ == "__main__":
    data = get_collection()

    for i, (doc_id, meta, doc) in enumerate(zip(data["ids"], data["metadatas"], data["documents"]), 1):
        print(f"{i}. ID: {doc_id}")
        print(f"   Metadata: {meta}")
        print(f"   Document: {doc[:100]}...\n")