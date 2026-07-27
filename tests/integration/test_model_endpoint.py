MODEL_INFO_FIELDS = [
    "model_name",
    "task",
    "class_names",
    "test_acc",
    "test_loss",
    "train_date",
    "training_epochs",
    "input_shape",
    "constructor_kwargs",
]


def test_model_info(client):
    client.app.state.primary_metadata = {key: None for key in MODEL_INFO_FIELDS}
    client.app.state.secondary_metadata = {key: None for key in MODEL_INFO_FIELDS}

    response = client.get("/model/info")

    assert response.status_code == 200

    body = response.json()

    for key in ["primary", "secondary"]:
        assert key in body
        for model_info_fields in MODEL_INFO_FIELDS:
            assert model_info_fields in body[key]
