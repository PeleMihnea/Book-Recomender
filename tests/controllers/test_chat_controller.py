# tests/controllers/test_chat_controller.py
import pytest
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from src.backend.app import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def load_env():
    load_dotenv()


def test_chat_endpoint_happy_path():
    payload = {"question": "Vreau o carte despre prietenie"}
    response = client.post("/api/chat", json=payload)

    # Check status code
    assert response.status_code == 200, response.text

    # Check response JSON structure
    data = response.json()
    assert "recommendation" in data
    assert "detailed_summary" in data
    assert isinstance(data["recommendation"], str)
    assert isinstance(data["detailed_summary"], str)


def test_chat_endpoint_empty_question():
    response = client.post("/api/chat", json={"question": "   "})

    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Question is required"
