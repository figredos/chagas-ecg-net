def test_is_live(client):
    response = client.get("/health/live")

    assert response.status_code == 200


def test_model_is_ready(client, mock_predictor):
    response = client.get("/health/ready")

    assert response.status_code == 200


def test_model_not_ready(client, mock_predictor):
    client.app.state.predictor = None
    response = client.get("/health/ready")

    print(response.content)
    assert response.status_code == 503
