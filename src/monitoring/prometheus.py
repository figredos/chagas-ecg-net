from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    name="request_count",
    documentation="Number of requests in API",
    labelnames=["endpoint", "status_code"],
)

ERROR_COUNT = Counter(
    name="error_counter",
    documentation="Number of resulting errors in API",
    labelnames=["endpoint", "status_code"],
)

PREDICTION_LATENCY = Histogram(
    name="prediction_latency",
    documentation="Latency for predictions",
    buckets=[10, 50, 100, 250, 500, 1000, 2500],
)

PREDICTION_CLASS_DISTRIBUTION = Counter(
    name="prediction_class_distribution",
    documentation="Distributions of predictions for each class",
    labelnames=["predicted_class"],
)
