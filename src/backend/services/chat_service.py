# src/backend/services/chat_service.py
from __future__ import annotations

import os
import json
import unicodedata
from pathlib import Path
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from src.backend.models.chat_models import ChatResponse
from src.backend.repositories.chroma_repo import ChromaRepository
from src.backend.tools.get_summary import SummaryTool

load_dotenv()


class ChatService:
    def __init__(self, anchors_path: str | None = None):
        # Configure OpenAI v1 client
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI()

        # Core dependencies
        self.repo = ChromaRepository()
        self.summary_tool = SummaryTool()

        # ---------------- Domain gating configuration ----------------
        # 1) Keywords (RO + EN). We normalize (remove accents) at runtime.
        self._book_keywords: set[str] = {
            # Romanian
            "carte", "cărți", "roman", "romane", "autor", "autori",
            "literatură", "literar", "gen", "genuri", "personaj", "personaje",
            "rezumat", "recomandare", "recomandări", "titlu", "lectură",
            "temă", "teme", "operă", "opere", "aventură", "aventuri", "fantasy", "istorie", "istoric", "science fiction", "sci-fi",
            "dramă", "mister", "thriller", "horror", "comedie", "romantic", "romance",
            "distopie", "utopie", "biografie", "autobiografie", "eseu", "poezie", "suspans",
            "dragoste", "război", "magie", "mitologie", "familie", "prietenie", "libertate",
            "călătorie", "căutare", "maturizare", "speranță", "curaj", "trădare", "morală",
            "societate", "supraviețuire", "filosofie",
            "poveste", "narațiune", "lectură", "lecturi", "narator", "conflict", "tematică",
            "final", "personaj principal", "personaje secundare", "scenariu", "cadru", "acțiune",
            "subiect", "titlu de carte", "recomandare literară", "critică literară",
            "bestseller", "clasic", "modern", "contemporan", "epic", "fabulă",
            "nuvelă", "povestire", "trilogie", "serie", "volum", "capitol",
            # Common verbs/variants
            "recomanzi", "recomand", "recomanda", "recomandati",
            "rezumati", "rezuma", "rezumatul", "citesc", "lectura",
            "genuri literare", "gen literar", "roman fantasy", "roman istoric",
            "roman science fiction", "roman polițist", "roman de dragoste",
            "carte de aventuri", "carte de mister", "carte de horror",
            # English (mixed inputs)
            "book", "books", "novel", "author", "authors", "genre", "genres",
            "character", "characters", "summary", "recommendation",
            "title", "reading", "theme", "themes",
            "war", "magic", "love", "friendship", "history", "adventure", "plot", "theme",
            "narrative", "ending", "main character", "genre", "fantasy novel", "classic",
            "mystery", "epic", "hero", "villain", "conflict", "resolution", "literary work",
        }
        # Pre-compute normalized (accent-free) keywords
        self._book_keywords_norm = {self._strip_accents(w) for w in self._book_keywords}

        # 2) Semantic anchors: in-domain vs out-of-domain
        self._anchors_book = [
            "Recomandări de cărți", "Rezumat de carte sau roman",
            "Informații despre autori și genuri literare",
            "Personaje, teme și subiecte literare",
            "Caut o carte potrivită intereselor mele",
            "Book recommendations", "Book or novel summary",
            "Information about authors and literary genres",
            "Characters, themes, and literary topics",
        ]
        self._anchors_non_book = [
            "Automobile, mașini, vehicule, condus, motoare",
            "Rețete de gătit, mâncare, bucătărie",
            "Prognoza meteo și temperaturi",
            "Programare, codare, inginerie software",
            "Recomandări de călătorie, zboruri și hoteluri",
            "Sport, fotbal, baschet, tenis",
            "Politică și alegeri",
        ]

        # 3) Thresholds
        self._threshold = 0.80            # strict semantic threshold without keywords
        self._margin = 0.08               # book score must exceed non-book by margin
        self._threshold_keywords = 0.70   # relaxed semantic threshold when keywords are present

        # 4) Cache anchor embeddings to disk
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        book_cache = cache_dir / "anchors_book.npy"
        non_book_cache = cache_dir / "anchors_non_book.npy"

        if book_cache.exists():
            self._book_vecs = np.load(book_cache)
        else:
            self._book_vecs = self._embed_texts(self._anchors_book)
            np.save(book_cache, self._book_vecs)

        if non_book_cache.exists():
            self._non_book_vecs = np.load(non_book_cache)
        else:
            self._non_book_vecs = self._embed_texts(self._anchors_non_book)
            np.save(non_book_cache, self._non_book_vecs)
        # ---------------------------------------------------------------------

    # ----------------------------- Embeddings -----------------------------
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using OpenAI embeddings and return an NxD matrix."""
        resp = self._client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        vecs = [d.embedding for d in resp.data]
        return np.asarray(vecs, dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-12
        return float(np.dot(a, b) / denom)

    def _max_cosine(self, v: np.ndarray, mat: np.ndarray) -> float:
        return max(self._cosine(v, row) for row in mat) if len(mat) else 0.0

    def _domain_scores(self, query: str) -> tuple[float, float]:
        """Return (book_score, non_book_score) via cosine to anchor matrices."""
        qv = self._embed_texts([query])[0]
        return self._max_cosine(qv, self._book_vecs), self._max_cosine(qv, self._non_book_vecs)

    # ----------------------------- Keywords ------------------------------
    @staticmethod
    def _strip_accents(s: str) -> str:
        """Lowercase and remove diacritics (ăâîșț → aasit)."""
        nfkd = unicodedata.normalize("NFD", s)
        return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower()

    def _has_book_keywords(self, text: str) -> bool:
        """Cheap allowlist gate using normalized keywords."""
        t = self._strip_accents(text)
        return any(k in t for k in self._book_keywords_norm)

    # ----------------------------- Moderation ----------------------------
    def moderate(self, text: str) -> bool:
        """
        Composite moderation:
        1) Safety moderation (OpenAI Moderations).
        2) Book-domain gate: keywords -> allow (or relaxed semantic), else strict semantic.
        Returns True if allowed, False if blocked.
        """
        # 1) Safety moderation
        try:
            resp = self._client.moderations.create(
                model="omni-moderation-latest",
                input=text
            )
            if resp.results[0].flagged:
                return False
        except Exception as e:
            # If moderation fails, choose your policy. Here we log and continue.
            print(f"[WARN] Moderation API failed: {e}")

        # 2) Domain gating
        if self._has_book_keywords(text):
            # Easiest: allow immediately (uncomment if you prefer this path)
            return True
            # Or: relaxed semantic check instead of immediate allow:
            # book, non_book = self._domain_scores(text)
            # return (book >= self._threshold_keywords) and (book >= non_book)

        # No keywords → strict semantic in-vs-out
        book, non_book = self._domain_scores(text)
        return (book >= self._threshold) and (book >= non_book + self._margin)

    # -------------------------- Prompt & Chat ----------------------------
    def _build_prompt(self, question: str, retrieved) -> str:
        """Build a compact context for the LLM from retrieved candidates."""
        context_lines = []
        for i, b in enumerate(retrieved, start=1):
            themes = ", ".join(b.themes)
            context_lines.append(
                f"{i}. Title: {b.title}\n   Themes: {themes}\n   Summary: {b.summary}"
            )
        context_block = "\n\n".join(context_lines)

        return (
            "You are a helpful book recommender.\n"
            "Given the user's request and the candidate books below, "
            "pick the single best title EXACTLY as written. "
            "Also give a short reasoning (2–3 sentences).\n\n"
            f"User request: {question}\n\n"
            f"Candidates:\n{context_block}\n\n"
            "Output strictly as JSON: {\"title\": \"...\", \"reasoning\": \"...\"}"
            "Give the answer in the same language as the question."
        )

    def _recommend(self, question: str, retrieved) -> Tuple[str, str]:
        prompt = self._build_prompt(question, retrieved)
        res = self._client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "Recomandă cărți doar dintre candidații furnizați."},
                {"role": "user", "content": prompt},
            ],
        )
        text = res.choices[0].message.content or ""

        try:
            data = json.loads(text)
            return data.get("title", ""), data.get("reasoning", "")
        except Exception:
            # Fallback: take top retrieved when parsing fails
            return (retrieved[0].title if retrieved else ""), "Fallback to top match."

    # ----------------------------- Public API ----------------------------
    def handle_chat(self, question: str) -> ChatResponse:
        # 1) Moderation (safety + domain)
        if not self.moderate(question):
            return ChatResponse(
                recommendation="",
                reasoning=("Acest asistent răspunde doar la întrebări despre cărți "
                           "(recomandări, rezumate, autori, genuri). Încearcă să reformulezi întrebarea în acest domeniu."),
                detailed_summary="",
            )

        # 2) Retrieval
        candidates = self.repo.search(question, k=3)
        if not candidates:
            return ChatResponse(
                recommendation="",
                reasoning="No relevant results found.",
                detailed_summary="",
            )

        # 3) LLM pick
        title, reasoning = self._recommend(question, candidates)

        # 4) Detailed summary via tool
        full_summary = self.summary_tool.get_summary_by_title(title)

        return ChatResponse(
            recommendation=title,
            reasoning=reasoning,
            detailed_summary=full_summary,
        )
