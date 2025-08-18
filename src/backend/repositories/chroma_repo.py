import os
from typing import List

import openai
import chromadb

from src.backend.models.chat_models import RetrievedBook

class ChromaRepository:
    def __init__(self):
        # Load API key and init OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        chromadb_dir = os.getenv("CHROMA_DIR")
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        # Initialize Chroma client & collection
        self.client = chromadb.PersistentClient(path=chromadb_dir)
        self.collection = self.client.get_collection("book_summaries")

    def _embed(self, text: str) -> List[float]:
        """Call OpenAI to embed a piece of text."""
        response = openai.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def search(self, query: str, k: int = 3) -> List[RetrievedBook]:
        """
        Embed the query, run a k-NN search in ChromaDB,
        and return the top-k retrieved books.
        :param query: The search query (e.g. a theme or keyword).
        :param k: Number of results to return.
        :return: List of RetrievedBook objects with title, summary, themes, and score.
        """
        vector = self._embed(query)
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )

        books: List[RetrievedBook] = []
        for meta, doc, dist in zip(
            results["metadatas"][0],
            results["documents"][0],
            results["distances"][0]
        ):
            raw = meta.get("themes", [])
            if isinstance(raw, list):
                theme_list = raw
            elif isinstance(raw, str):
                theme_list = [t.strip() for t in raw.split(",") if t.strip()]
            else:
                theme_list = []

            books.append(RetrievedBook(
                title=meta["title"],
                summary=doc,
                themes=theme_list,
                score=dist
            ))
        return books
