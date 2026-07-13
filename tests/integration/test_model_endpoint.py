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
    client.app.state.metadata = {key: None for key in MODEL_INFO_FIELDS}

    response = client.get("/model/info")

    assert response.status_code == 200

    body = response.json()

    for key in MODEL_INFO_FIELDS:
        assert key in body
