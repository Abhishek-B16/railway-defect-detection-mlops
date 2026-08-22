# MLflow Model Registry Documentation

## Why Model Registry is Needed
In an MLOps lifecycle, you often train dozens or hundreds of models across multiple experiments. The **MLflow Model Registry** acts as a centralized repository to store, version, and manage the lifecycle of these models. 
Instead of searching through local folders for `best.pt` weights and trying to remember which dataset version produced them, the Model Registry securely tracks:
- Model versions and iterations
- Lineage (exactly which training run and dataset produced the model)
- Lifecycle stages (e.g., Staging, Production, Candidate)
- Metadata and performance metrics

## Registered Model Details
- **Registered Model Name:** `RailGuard-Detector`
- **Version:** Version 1
- **Alias:** `candidate` (Marks this model for potential future promotion)
- **Current Status:** Registered successfully. It is currently a candidate model, not yet marked for production.

## Lineage & Evaluation
- **Source MLflow Run:** The run that logged this specific model (Refer to `mlruns/` SQLite database for exact UUID).
- **Dataset Version:** `dataset_clean` (tracked via `dataset_clean.dvc`)
- **Evaluation Metrics:**
  - mAP50: 0.578
  - Classes covered: `bolts, crack, flaking, joints, sheling, spalling`

## How to Load the Model
Because the model is securely registered in MLflow, any downstream application (like a FastAPI server) can load it dynamically using its registry name and alias without needing to hardcode local file paths:

```python
import mlflow

# Setup tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load the candidate model dynamically
model_uri = "models:/RailGuard-Detector@candidate"
loaded_model = mlflow.pyfunc.load_model(model_uri)
```
