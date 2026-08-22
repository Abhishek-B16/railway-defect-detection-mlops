import os
import glob
import mlflow
from mlflow.tracking import MlflowClient

def register_model():
    # 1. Setup MLflow client
    # To support Model Registry, we must use a database backend.
    # We will initialize a sqlite database.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("RailGuard")
    client = MlflowClient()

    # 2. Find the best model from the previous YOLO run
    best_model_path = r"c:\Users\ABHISHEK\ML project\railway-defect-mlops\runs\detect\runs\detect\train\weights\best.pt"
    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}")
        return

    # 3. Create a new run to log the model properly into the DB backend
    print("Creating registry entry in MLflow...")
    with mlflow.start_run(run_name="Model_Registration") as run:
        run_id = run.info.run_id
        
        # Log the model artifact. We log it as a generic artifact since YOLOv8
        # integrates differently with mlflow.pyfunc
        mlflow.log_artifact(best_model_path, artifact_path="model")
        
        # Define the model URI
        model_uri = f"runs:/{run_id}/model/best.pt"
        
        # 4. Register the model using create_model_version directly
        model_name = "RailGuard-Detector"
        print(f"Registering model: {model_name}")
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass # already exists
            
        mv = client.create_model_version(
            name=model_name, 
            source=model_uri, 
            run_id=run_id
        )
        print(f"Successfully registered model '{model_name}' version {mv.version}")

        # 5. Add Description
        client.update_registered_model(
            name=model_name,
            description="RailGuard is a railway track defect detection model that detects six defect categories using YOLOv8 object detection."
        )

        # 6. Add Tags
        tags = {
            "project": "RailGuard",
            "model_type": "YOLOv8n",
            "task": "railway_defect_detection",
            "dataset": "dataset_clean",
            "dataset_version": "dataset_clean.dvc",
            "test_map50": "0.578",
            "classes": "bolts, crack, flaking, joints, sheling, spalling"
        }
        for key, value in tags.items():
            client.set_model_version_tag(
                name=model_name,
                version=mv.version,
                key=key,
                value=value
            )
            
        # 7. Set Alias
        client.set_registered_model_alias(
            name=model_name,
            alias="candidate",
            version=mv.version
        )
        print("Successfully applied alias 'candidate' and metadata tags.")
        
        # Output report data
        print("\n--- REGISTRATION REPORT ---")
        print(f"Registered Model Name: {model_name}")
        print(f"Model Version: {mv.version}")
        print(f"MLflow Run ID: {run_id}")
        print(f"Model URI: {model_uri}")
        print(f"Alias: candidate")
        print(f"Test Metrics: mAP50 = 0.578")
        print("Status: Registration Succeeded")

if __name__ == "__main__":
    register_model()
