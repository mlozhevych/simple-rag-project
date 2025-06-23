import pytest
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qdrant_client")

@pytest.fixture(scope="module")
def qdrant_client():
    return QdrantClient(host="localhost", port=6333)

@pytest.fixture(scope="module")
def embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def test_qdrant_search(qdrant_client, embedding_model):
    query = "What does the API First Principle require in terms of development order?"
    vector = embedding_model.encode(query)

    results = qdrant_client.search(
        collection_name="documents",
        query_vector=vector,      # у старому клієнті це працює
        limit=3,
        with_payload=True,
    )

    assert results, "Qdrant нічого не знайшов"

    for point in results:
        payload = point.payload
        assert "text" in payload, "У payload відсутній ключ 'text'."
        assert ("source" in payload) or ("file_name" in payload), "Немає метаданих про файл."

    # виводимо результати для наочного перегляду
    for p in results:
        print(f"\nScore {p.score:.4f} | {p.payload.get('source') or p.payload.get('file_name')}")
        print(p.payload["text"][:200], "...\n---")