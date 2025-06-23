import os

import pytest
from qdrant_client import QdrantClient
from simple_rag_project.app import app as flask_app
from simple_rag_project.create_embeddings import run_indexing_pipeline


# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# --- Fixture for setting up the test database ---

@pytest.fixture(scope="session")
def qdrant_test_collection():
    """
    Session-scoped fixture to set up and tear down a test Qdrant collection.

    This runs once per test session:
    1. Sets up a clean, populated Qdrant collection for testing.
    2. Yields the collection name to the tests.
    3. Deletes the collection after all tests are done.
    """
    # Use environment variables, especially in CI
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    test_collection_name = "test_collection_ci"

    # Path to test data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'simple_rag_project', 'data')

    # Call the refactored function to populate the test database
    print("\n--- Setting up test Qdrant collection ---")
    run_indexing_pipeline(
        host=qdrant_host,
        port=qdrant_port,
        collection=test_collection_name,
        data_path=data_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Hardcode for consistency
        chunk_size=1000,
        chunk_overlap=100,
    )

    # Yield the collection name to the tests
    yield test_collection_name

    # --- Teardown ---
    print(f"\n--- Tearing down test Qdrant collection: {test_collection_name} ---")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    try:
        client.delete_collection(collection_name=test_collection_name)
    except Exception as e:
        print(f"Could not delete collection {test_collection_name}: {e}")


# --- Fixture for the Flask test client ---

@pytest.fixture(scope="module")
def test_client(qdrant_test_collection):
    """
    Module-scoped fixture to create a Flask test client.
    It depends on `qdrant_test_collection` to ensure the DB is ready.
    """
    # Configure the app for testing
    flask_app.config.update({
        "TESTING": True,
        # IMPORTANT: Tell the app to use our temporary test collection!
        "QDRANT_COLLECTION": qdrant_test_collection
    })

    with flask_app.test_client() as client:
        yield client
