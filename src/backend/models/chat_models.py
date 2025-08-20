from pydantic import BaseModel
from typing import List, Optional

class RetrievedBook(BaseModel):
    title: str
    summary: str
    themes: List[str]
    score: float

class ChatRequest(BaseModel):
    question: str
    voice_mode: Optional[bool] = False
    image: Optional[bool] = False

class ChatResponse(BaseModel):
    recommendation: str
    reasoning: str
    detailed_summary: str
    audio_url: Optional[str] = None
    image_url: Optional[str] = None