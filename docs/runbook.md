# Chagas-Ecg-Net Runbook

- [Chagas-Ecg-Net Runbook](#chagas-ecg-net-runbook)
  - [Adding new model version](#adding-new-model-version)
    - [Model versioning](#model-versioning)
    - [Changing API's model version](#changing-apis-model-version)
  - [Redeploying cloud run](#redeploying-cloud-run)

## Adding new model version

To add a new model version, use the scripts in `scripts/training`, and can be executed as shown bellow.

```bash
python3 -m scripts.training.train_{model_name}
```

- **Example: Training `grouped_lead_cnn`**

```bash
python3 -m scripts.training.train_grouped_lead_cnn
```

These scripts use config files from `configs` for variable injection, through the `hydra` package. Each model config is composed of 2 files:

- `configs/train_{model_name}.yaml`: Contains info such as the model's name, the task the model is being trained on, the seed to use, and hydra's base directory.
- `configs/{model_name}/{model_name.yaml}`:
  - Hyperparameters for:
    - Data, such as train/test size, and class names.
    - Training, such number of epochs, learning rate, scheduler settings, etc.
    - Model, such as batch size, and constructor kwargs.
  - Paths for checkpoints and logs.
  - MLflow settings.
  - Training progress output source (cli or notebook).

Each file can be altered for different experiments, either by modifying the file before runtime, or at runtime, by setting values through cli:

```bash
python3 -m scripts.training.train_{model_name} {config}={new_value}
```

- **Example: Training `cnn_bert`, and overwriting notebook config**

```bash
python3 -m scripts.training.train_grouped_lead_cnn cnn_bert.notebook=True
```

### Model versioning

Each config file has a specific model version config under `{model_name}.mlflow.version`, that is default `null`. This setting can be overwritten to specify a model version, which overwrites any existing version of the same name.

If no changes are made and the setting is still null at training time, the logging of the new version is set to $V(n+1)$, where $n$ is the latest version's number.

At the end of training each model's checkpoint is turned into a new model version and stored, alongside its metadata into the `/model_registry` folder, under the model's name and version.

### Changing API's model version

All of the API's settings are injected through the `src/api/config.py` file, that uses a specialization of the `BaseSettings` class from `pydantic`. This class imports settings such as `REGISTRY_ROOT`, `MODEL_NAME`, `MODEL_VERSION`, `DEVICE`, `CORS_ORIGIN`, and `MAX_UPLOAD_BYTES`, from a `.env` in the project's root.

To adjust which model the API is using, just adjust the `MODEL_NAME` setting to the one you would like to use. Same goes for `MODEL_VERSION`, which needs to exist within `{REGISTRY_ROOT}/{MODEL_NAME}`, otherwise generate an error.

## Redeploying cloud run

**_Cloud deployment is WIP_**
