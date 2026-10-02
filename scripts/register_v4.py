import mlflow
from mlflow.tracking import MlflowClient
import os
import getpass

def register_v4():
    print("To authorize DagsHub, please paste your Access Token below:")
    token = getpass.getpass("Token (it will be invisible as you type): ")
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Abhishek-B16"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token.strip()
    
    mlflow.set_tracking_uri("https://dagshub.com/Abhishek-B16/railway-defect-detection-mlops.mlflow")
    
    model_name = "RailGuard-Detector"
    # Update to point to the new V4 weights!
    model_path = r"runs\detect\runs\train\v4_experiment\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return

    print("Uploading V4 weights to DagsHub Model Registry...")
    with mlflow.start_run() as run:
        # 1. Upload the weights to MLflow
        mlflow.log_artifact(model_path, "model")
        
        # 2. Register it as a new version using create_model_version directly
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass # Already exists
            
        model_uri = f"runs:/{run.info.run_id}/model/best.pt"
        model_version = client.create_model_version(
            name=model_name, 
            source=model_uri, 
            run_id=run.info.run_id
        )
        
        # 3. Promote it to production by updating the @candidate alias
        client = MlflowClient()
        client.set_registered_model_alias(model_name, "candidate", model_version.version)
        
        print(f"SUCCESS! Registered {model_name} Version {model_version.version}!")
        print(f"The Render Production API (@candidate) will now use Version {model_version.version} (The V4 Model)!")

if __name__ == "__main__":
    register_v4()
