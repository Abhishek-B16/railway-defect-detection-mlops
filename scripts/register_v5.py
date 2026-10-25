import mlflow
import os
import getpass

def register_v5():
    print("To authorize DagsHub, please paste your Access Token below:")
    token = input("Token: ")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Abhishek-B16"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token.strip()
    
    mlflow.set_tracking_uri("https://dagshub.com/Abhishek-B16/railway-defect-detection-mlops.mlflow")
    client = mlflow.tracking.MlflowClient()
    
    # 1. Upload the V5 model weights to MLflow
    best_pt_path = r"runs\detect\runs\train\v5_experiment-2\weights\best.pt"
    if not os.path.exists(best_pt_path):
        print(f"Error: Could not find V5 weights at {best_pt_path}")
        return

    with mlflow.start_run(run_name="V5_YOLO_Large_HD_Registry") as run:
        print("\n⏳ WARNING: Uploading the 87 MB Version 5 AI Brain to DagsHub...")
        print("⏳ This usually takes 1 to 3 minutes depending on your Wi-Fi.")
        print("⏳ PLEASE WAIT. DO NOT CLOSE THIS TERMINAL...\n")
        
        mlflow.log_artifact(best_pt_path, "model_weights")
        model_uri = f"runs:/{run.info.run_id}/model_weights"
        
        print("✅ 87MB Upload Complete! Registering the model...")
        # 2. Register it as a new version
        model_name = "RailGuard-Detector"
        mv = mlflow.register_model(model_uri, model_name)
        print(f"Registered V5 as Version {mv.version} in MLflow!")
        
        # 3. Move the @candidate tag to V5!
        client.set_registered_model_alias(model_name, "candidate", mv.version)
        print("✅ Moved the @candidate tag to V5! The API will now download V5!")

if __name__ == "__main__":
    register_v5()
