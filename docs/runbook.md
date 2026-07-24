# Chagas-Ecg-Net Runbook

- [Chagas-Ecg-Net Runbook](#chagas-ecg-net-runbook)
  - [Adding new model version](#adding-new-model-version)
    - [Model versioning](#model-versioning)
    - [Changing API's model version](#changing-apis-model-version)
  - [Redeploying cloud run](#redeploying-cloud-run)
  - [Drift Monitoring](#drift-monitoring)
    - [Bias limitation](#bias-limitation)
    - [What to do on Drift detection](#what-to-do-on-drift-detection)
    - [Other limitations](#other-limitations)

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

Cloud Run redeployment involves 2 steps, re-building and pushing the docker image and deploying the image to cloud_run:

```bash
# 1. Build for linux/amd64 and push
docker buildx build --platform linux/amd64 \
    -t europe-west2-docker.pkg.dev/PROJECT_ID/chagas-ecg-net/chagas-ecg-net:latest \
    --push .

# 2. Deploy to Cloud Run
gcloud run deploy chagas-ecg-net \
    --image=europe-west2-docker.pkg.dev/PROJECT_ID/chagas-ecg-net/chagas-ecg-net:latest \
    --region=europe-west2 \
    --allow-unauthenticated \
    --port=8000 \
    --memory=2Gi \
    --set-env-vars="..."
```

`--set-env-vars` should be filled with pairs of keys and values from the `.env` variables.

## Drift Monitoring

Drift monitoring is automatically performed through the `/feedback` endpoint. Once the number of feedback entries is divisible by the value of the config `drift_feedback_window`, the `run_drift_check` function is executed and produces an output file with the following fields:

- **Drifted**: A boolean flag that indicates whether the model has drifted or not.
- **Disagreement_rate**: Measures the rolling disagreement rate between model predictions and confirmed user corrections. This measurement was favoured over class distribution due to datasets being from different regions with different Chagas prevalence rates, making distribution comparison ambiguous. And the training and testing datasets being balanced.
- **Sample count**: Number of analysed samples
- **Insufficient data**: A boolean flag that indicates whether there was enough data for a valid analysis.

### Bias limitation

Although the intended use of the API is for specialized medical doctors to have a more nuanced approach to asking for exams, there should still be some bias due to Voluntary feedback. Users are more likely to correct confident errors than ambiguous ones, meaning the true error rate is underestimated.

### What to do on Drift detection

If there is any model drift detected through the feedback information, there are other solutions/explanations besides model retraining, and checking them for inconsistencies and errors is very important.

- Inspecting recent feedback.
- Comparing per class errors.
- Verify the drift signal is real and not an artifact of a small or unrepresentative feedback window

Only after these checks failed should retraining be chosen.

### Other limitations

This performance monitoring is not the most comprehensive, but a mere step in the completeness of the API. Full performance monitoring requires a labelled production dataset, and that falls far from the current iteration of the project.
