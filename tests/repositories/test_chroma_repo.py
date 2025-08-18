import pytest
from dotenv import load_dotenv
from src.backend.repositories.chroma_repo import ChromaRepository


@pytest.fixture(autouse=True)
def load_env():
    load_dotenv()


def test_search_returns_books(capfd):
    repo = ChromaRepository()
    # Query on a theme we know exists
    results = repo.search("prietenie", k=2)

    # Print results for debug/logging
    print("\nRetrieved books:")
    for book in results:
        print(f"- {book.title} (score={book.score:.4f})")
        print(f"  Themes: {book.themes}")
        print(f"  Summary: {book.summary[:100]}...\n")

    # Capture pytest output so it shows in logs
    out, _ = capfd.readouterr()
    assert "Retrieved books:" in out

    # Assertions
    assert isinstance(results, list)
    assert len(results) > 0

    returned_titles = [book.title for book in results]
    assert "The Hobbit" in returned_titles  # prietenie should match The Hobbit

    for book in results:
        assert isinstance(book.title, str)
        assert isinstance(book.summary, str)
        assert isinstance(book.themes, list)
        assert isinstance(book.score, float)
