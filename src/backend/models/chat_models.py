from pydantic import BaseModel
from typing import List, Dict

class RetrievedBook(BaseModel):
    title: str
    summary: str
    themes: List[str]
    score: float
