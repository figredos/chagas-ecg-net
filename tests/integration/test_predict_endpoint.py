def test_happy_path(client, mock_predictor):

    record_path = "tests/fixtures/"

    with open(record_path + "sample.dat", "rb") as dat:
        dat_bytes = dat.read()

    with open(record_path + "sample.hea", "rb") as hea:
        hea_bytes = hea.read()

    response = client.post(
        "/predict",
        files={
            "file": (
                "sample.dat",
                dat_bytes,
                "application/octet-stream",
            ),
            "hea_file": (
                "sample.hea",
                hea_bytes,
                "application/octet-stream",
            ),
        },
    )
    body = response.json()

    assert response.status_code == 200

    assert isinstance(body["predicted_class"], str)
    assert isinstance(body["confidence"], float)
    assert isinstance(body["class_probabilities"], dict)
    assert isinstance(body["model_name"], str)
    assert isinstance(body["model_version"], str)
    assert isinstance(body["inference_time_ms"], float)

    assert (
        body["predicted_class"] == mock_predictor.route.return_value[1].predicted_class
    )


def test_oversized_file(client, mock_predictor):
    max_bytes = 52_428_800

    dat_bytes = b"0" * (max_bytes + 1)
    hea_bytes = b"0"

    response = client.post(
        "/predict",
        files={
            "file": (
                "sample.dat",
                dat_bytes,
                "application/octet-stream",
            ),
            "hea_file": (
                "sample.hea",
                hea_bytes,
                "application/octet-stream",
            ),
        },
    )

    assert response.status_code == 413


def test_unsuported_format(client, mock_predictor):
    dat_bytes = b"0"
    hea_bytes = b"0"

    file_bytes = b"0"

    response_invalid_dat = client.post(
        "/predict",
        files={
            "file": (
                "sample.txt",
                file_bytes,
                "application/octet-stream",
            ),
            "hea_file": (
                "sample.hea",
                hea_bytes,
                "application/octet-stream",
            ),
        },
    )

    assert response_invalid_dat.status_code == 415

    response_invalid_hea = client.post(
        "/predict",
        files={
            "file": (
                "sample.dat",
                dat_bytes,
                "application/octet-stream",
            ),
            "hea_file": (
                "sample.txt",
                file_bytes,
                "application/octet-stream",
            ),
        },
    )

    assert response_invalid_hea.status_code == 415

    response_invalid_both = client.post(
        "/predict",
        files={
            "file": (
                "sample.txt",
                file_bytes,
                "application/octet-stream",
            ),
            "hea_file": (
                "sample.txt",
                file_bytes,
                "application/octet-stream",
            ),
        },
    )

    assert response_invalid_both.status_code == 415
