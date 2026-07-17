METRICS_FIELDS = [
    "request_count",
    "error_count",
    "prediction_latency",
]


def test_metrics(client):
    response = client.get("/metrics")

    assert response.status_code == 200

    body = response.text

    for key in METRICS_FIELDS:
        assert key in body
