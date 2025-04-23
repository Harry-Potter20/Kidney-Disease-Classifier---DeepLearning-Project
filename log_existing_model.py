import os
import mlflow
import mlflow.keras
import tensorflow as tf
from dotenv import load_dotenv
import yaml
import json
from mlflow.tracking import MlflowClient

# ✅ 1. Load the .env file
load_dotenv()

# ✅ 2. Set MLflow config
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# ✅ 3. Optional authentication for DagsHub
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if username and password:
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

# ✅ 4. Load existing model
model_path = "artifacts/training/model.h5"
model = tf.keras.models.load_model(model_path)

# ✅ 5. Load config and scores
with open("params.yaml") as f:
    params = yaml.safe_load(f)

scores = {}
if os.path.exists("scores.json"):
    with open("scores.json") as f:
        scores = json.load(f)

# ✅ 6. Start MLflow logging
mlflow.set_experiment("KidneyDiseaseEvaluation")

with mlflow.start_run(run_name="Manual Upload to DagsHub") as run:
    run_id = run.info.run_id

    # Log parameters
    mlflow.log_param("model_path", model_path)
    mlflow.log_param("image_size", params["IMAGE_SIZE"])
    mlflow.log_param("batch_size", params["BATCH_SIZE"])
    mlflow.log_param("freeze_till", params["FREEZE_TILL"])

    # Log metrics
    for k, v in scores.items():
        mlflow.log_metric(k, v)

    # Log model
    mlflow.keras.log_model(model, artifact_path="model")
    print("✅ Model logged to MLflow (DagsHub)")

    # ✅ Register the model
    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name="KidneyDiseaseClassifier"
    )
    print(f"✅ Model registered as '{result.name}' (version: {result.version})")

    # ✅ Optionally promote to staging or production
    client = MlflowClient()
    client.transition_model_version_stage(
        name=result.name,
        version=result.version,
        stage="Staging"
    )
    print(f"🚀 Model transitioned to 'Staging' stage.")
