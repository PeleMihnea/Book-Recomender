# src/backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.backend.controllers.chat_controller import router as chat_router

app = FastAPI(title="Book Recommender")

# CORS: allow Streamlit on localhost (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

@app.get("/health")
def health():
    return {"status": "ok"}
