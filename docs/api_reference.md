# Chagas-ECG-Net API Reference

- [Chagas-ECG-Net API Reference](#chagas-ecg-net-api-reference)
  - [Endpoints](#endpoints)
    - [`POST /predict`](#post-predict)
    - [`POST /feedback`](#post-feedback)
    - [`GET /health/live`](#get-healthlive)
    - [`GET /health/ready`](#get-healthready)
    - [`GET /metrics`](#get-metrics)
    - [`GET /model/info`](#get-modelinfo)

## Endpoints

### `POST /predict`

- Purpose
  - Runs inference on a WFDB ECG file pair.

- Request parameters
  - ```
    JSON: Multipart/form-data

    file: .dat file

    hea_file: .hea file
    ```

- Response Schemas
  - ```json
      {
      "predicted_class": "string",
      "confidence": float,
      "class_probabilities": {
      "chagas": float,
      "non-chagas": float
      },
      "model_name": "string",
      "model_version": "string",
      "inference_time_ms": float
      }
    ```

- Status codes
  - `200`: Successful response.
  - `400`: .dat or .hea files are malformed.
  - `413`: .dat contents are too large.
  - `415`: Unsupported file format for either .dat or .hea field.
  - `422`: Pydantic validation error.
  - `500`: Internal server error.

- Example `curl`
  - ```bash
        curl -X 'POST' \
            'http://127.0.0.1:8000/predict' \
            -H 'accept: application/json' \
            -H 'Content-Type: multipart/form-data' \
            -F 'file=@01000_hr.dat' \
            -F 'hea_file=@01000_hr.hea'
    ```

---

### `POST /feedback`

- Purpose
  - Submit a correction to a prediction.

- Request parameters
  - ```json
    {
    "predicted_class": "string",
    "true_class": "string",
    "confidence_score": float
    }
    ```
- Response Schemas
  - Only return code 202.

- Status codes
  - `202`: JSON request received.
  - `422`: Pydantic validation error.

- Example `curl`
  - ```bash
        curl -X 'POST' \
            'http://127.0.0.1:8000/feedback' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{
            "predicted_class": "string",
            "true_class": "string",
            "confidence_score": 0
        }'
    ```

---

### `GET /health/live`

- Purpose
  - Liveness probe, checks if app is up.

- Request parameters
  - No parameters.

- Response Schemas
  - ```json
    {
    "status": status_code (int),
    "details": "string"
    }
    ```

- Status codes
  - `200`: Successful response.

- Example `curl`
  - ```bash
        curl -X 'GET' \
            'http://127.0.0.1:8000/health/live' \
            -H 'accept: application/json'
    ```

---

### `GET /health/ready`

- Purpose
  - Readiness probe, checks if model is loaded.

- Request parameters
  - No parameters.

- Response Schemas
  - ```json
    {
    "status": status_code (int),
    "details": "string"
    }
    ```

- Status codes
  - `200`: Successful response.
  - `503`: Model instance still loading.

- Example `curl`
  - ```bash
        curl -X 'GET' \
            'http://127.0.0.1:8000/health/ready' \
            -H 'accept: application/json'
    ```

---

### `GET /metrics`

- Purpose
  - Request counts, error counts, latency.

- Request parameters
  - No parameters.

- Response Schemas
  - ```json
    {
      "request_count": 7,
      "error_count": 0,
      "total_latency_ms": 22.775956997065805,
      "avg_pred_latency_ms": 3.253708142437972,
      "total_uptime_seconds": 1942.0610558986664
    }
    ```

- Status codes
  - `200`: Successful response

- Example `curl`
  - ```bash
        curl -X 'GET' \
            'http://127.0.0.1:8000/metrics' \
            -H 'accept: application/json'
    ```

---

### `GET /model/info`

- Purpose
  - Metadata for currently loaded model.

- Request parameters
  - No parameters.

- Response Schemas
  - ```json
    {
      "model_name": "string",
      "task": "string",
      "class_names": ["string"],
      "test_acc": float,
      "test_loss": float,
      "train_date": "string",
      "training_epochs": int,
      "input_shape": [int],
      "constructor_kwargs": {"string": any}
    }
    ```

- Status codes
  - `200`: Successful response.
  - `503`: Model's info hasn't been loaded yet.

- Example `curl`
  - ```bash
        curl -X 'GET' \
            'http://127.0.0.1:8000/model/info' \
            -H 'accept: application/json'
    ```
