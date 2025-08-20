# tests/tools/test_get_summary.py
import os
import pytest
from dotenv import load_dotenv
from src.backend.tools.get_summary import SummaryTool

@pytest.fixture(autouse=True)
def _env():
    load_dotenv()

def test_get_summary_by_title_returns_text():
    tool = SummaryTool()
    s = tool.get_summary_by_title("The Hobbit")

    assert isinstance(s, str)
    assert len(s) > 0
    assert "Bilbo" in s or "hobbit" in s.lower()