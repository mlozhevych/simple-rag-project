import os
import warnings

import pytest
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=DeprecationWarning, module="qdrant_client")


@pytest.fixture(scope="module")
def qdrant_client():
    return QdrantClient(host="localhost", port=6333)


@pytest.fixture(scope="module")
def embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def test_qdrant_search(qdrant_client, embedding_model, qdrant_test_collection):
    """
    Tests that we can connect to Qdrant and find relevant documents
    in the collection created by the fixture.
    """
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Use the collection name provided by the fixture
    collection_name = qdrant_test_collection

    # Check that the collection exists
    collections_response = client.get_collections()
    assert collection_name in [c.name for c in collections_response.collections]

    # Perform a search
    query = "What are the general principles of REST API design?"
    hits = client.search(
        collection_name=collection_name,
        query_vector=model.encode(query).tolist(),
        limit=3,
    )

    # Check that search returned results
    assert len(hits) > 0, "Search should return at least one result."
    assert "text" in hits[0].payload, "Result payload should contain the original text."
    print(f"Found {len(hits)} results.")
