# src/backend/tools/get_summary.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import chromadb


class SummaryTool:
    """
    Summary lookup backed by ChromaDB.
    Assumes:
      - Collection name: 'book_summaries'
      - Title is stored in metadata: {"title": "<exact title>"}
      - The summary text is stored as the document itself.
    """

    def __init__(self, chroma_path: Optional[str] = None, collection_name: str = "book_summaries"):
        # Resolve Chroma persistence directory
        # 1) env override (CHROMA_DIR), 2) ctor arg, 3) default to project_root/data/embeddings
        if chroma_path is None:
            chroma_path = os.getenv("CHROMA_DIR")

        if chroma_path is None:
            # Compute project root from this file: tools -> backend -> src -> <root>
            project_root = Path(__file__).resolve().parents[3]
            chroma_path = str(project_root / "data" / "embeddings")

        # Initialize Chroma client and collection
        self._client = chromadb.PersistentClient(path=chroma_path)
        # Use get_or_create to be resilient; it will open if exists
        self._col = self._client.get_or_create_collection(name=collection_name)

    def get_summary_by_title(self, title: str) -> str:
        """
        Fetch the full summary by exact title from ChromaDB metadata.
        Returns empty string if no match is found.
        """
        # Filter by metadata exact match on title
        res = self._col.get(where={"title": title}, include=["documents", "metadatas"])
        docs = res.get("documents") or []
        # documents is a list-of-lists in some versions; normalize
        if docs and isinstance(docs[0], list):
            docs = docs[0]
        return docs[0] if docs else ""
