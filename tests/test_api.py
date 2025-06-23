def test_health_endpoint(test_client):
    """Tests the /health endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "OK"}


# def test_ask_endpoint_success(test_client):
#     """
#     Tests the /ask endpoint with a valid query.
#     This test implicitly depends on `qdrant_test_collection` because `test_client` does.
#     """
#     response = test_client.post("/ask", json={"question": "What is REST?"})
#
#     assert response.status_code == 200
#     data = response.json
#     assert "answer" in data
#     assert len(data["answer"]) > 10  # Check for a meaningful answer
#     print(f"API Answer: {data['answer'][:100]}...")


def test_ask_endpoint_no_query(test_client):
    """Tests the /ask endpoint with a missing query."""
    response = test_client.post("/ask", json={})
    assert response.status_code == 400
    assert "error" in response.json
    assert response.json["error"] == "Invalid request. 'question' field is required."
