# src/backend/controllers/chat_controller.py
from fastapi import APIRouter, HTTPException
from src.backend.models.chat_models import ChatRequest, ChatResponse
from src.backend.services.chat_service import ChatService

router = APIRouter(prefix="/api", tags=["chat"])
_service = ChatService()

@router.get("/debug/mod")
def debug_mod(q: str):
    """Return the internal moderation decision + signals."""
    # expose internal checks (dev only)
    has_kw = _service._has_book_keywords(q)
    book, non_book = _service._domain_scores(q)
    decision = _service.moderate(q)
    return {
        "query": q,
        "has_keywords": has_kw,
        "book_score": round(book, 4),
        "non_book_score": round(non_book, 4),
        "threshold": _service._threshold,
        "margin": _service._margin,
        "moderate_decision": decision,
    }


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    HTTP entrypoint for chat. Validates input, enforces moderation/domain gating,
    then delegates to ChatService for retrieval + LLM + tool.
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is required")

    # Enforce safety + domain gating (books-only) BEFORE generating a response
    if not _service.moderate(q):
        raise HTTPException(
            status_code=422,
            detail=("Acest asistent răspunde doar la întrebări despre cărți "
                    "(recomandări, rezumate, autori, genuri).")
        )

    return _service.handle_chat(q)
