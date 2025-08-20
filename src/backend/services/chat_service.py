import os
import re
from openai import OpenAI
from typing import Tuple
from dotenv import load_dotenv

from src.backend.repositories.chroma_repo import ChromaRepository
from src.backend.models.chat_models import ChatResponse
from src.backend.tools.get_summary import SummaryTool

# Load environment variables
load_dotenv()

class ChatService:
    def __init__(self):
        OpenAI.api_key = os.getenv("OPENAI_API_KEY", "")
        if not OpenAI.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        self.repo = ChromaRepository()
        self.summary_tool = SummaryTool()
        self._client = OpenAI()

    # ---------- moderation ----------

    def moderate(self, text: str) -> bool:
        """Return True if safe, False if flagged."""
        try:
            resp = self._client.moderations.create(
                model="omni-moderation-latest",
                input=text
            )
            # v1: attribute access
            return not resp.results[0].flagged
        except Exception as e:
            print(f"[WARN] Moderation API failed: {e}")
            # choose your policy: allow (True) or block (False)
            return True

    # ---------- prompt building ----------
    def _build_prompt(self, question: str, retrieved) -> str:
        """
        Build context for GPT with retrieved candidates.
        """
        context_lines = []
        for i, b in enumerate(retrieved, start=1):
            context_lines.append(
                f"{i}. Title: {b.title}\n   Themes: {', '.join(b.themes)}\n   Summary: {b.summary}"
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
        )

    # ---------- LLM call ----------
    def _recommend(self, question: str, retrieved) -> tuple[str, str]:
        prompt = self._build_prompt(question, retrieved)

        res = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Recomanzi cărți doar din candidații furnizați.."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        text = res.choices[0].message.content or ""

        # parse the JSON the model returns
        import json
        try:
            data = json.loads(text)
            return data.get("title", ""), data.get("reasoning", "")
        except Exception:
            # fallback to top retrieved
            return (retrieved[0].title if retrieved else ""), "Fallback to top match."

    # ---------- public entry ----------
    def handle_chat(self, question: str) -> ChatResponse:
        # 1. Moderation
        if not self.moderate(question):
            return ChatResponse(
                recommendation="",
                reasoning="The message contained inappropriate language.",
                detailed_summary="",
            )

        # 2. Retrieve candidates
        candidates = self.repo.search(question, k=3)
        if not candidates:
            return ChatResponse(
                recommendation="",
                reasoning="No relevant results found.",
                detailed_summary="",
            )

        # 3. LLM picks best candidate
        title, reasoning = self._recommend(question, candidates)

        # 4. Get full summary by title
        full_summary = self.summary_tool.get_summary_by_title(title)

        return ChatResponse(
            recommendation=title,
            reasoning=reasoning,
            detailed_summary=full_summary,
        )
