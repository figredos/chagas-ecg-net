METRICS_FIELDS = [
    "request_count",
    "error_count",
    "total_latency_ms",
    "avg_pred_latency_ms",
    "total_uptime_seconds",
]


def test_metrics(client):
    response = client.get("/metrics")

    assert response.status_code == 200

    body = response.json()

    for key in METRICS_FIELDS:
        assert key in body
