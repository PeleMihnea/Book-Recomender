# src/backend/controllers/chat_controller.py
from fastapi import APIRouter, HTTPException
from src.backend.models.chat_models import ChatRequest, ChatResponse
from src.backend.services.chat_service import ChatService

router = APIRouter(prefix="/api", tags=["chat"])
_service = ChatService()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    HTTP entrypoint for chat. Delegates to ChatService.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    resp = _service.handle_chat(req.question.strip())
    return resp
