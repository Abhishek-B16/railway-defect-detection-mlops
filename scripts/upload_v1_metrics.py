import os
import pandas as pd
import mlflow
import getpass

def upload_v1_metrics():
    # 1. Setup MLflow to point to DagsHub
    token = getpass.getpass("To authorize DagsHub, please paste your Access Token below:\n")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Abhishek-B16"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token.strip()
    
    mlflow.set_tracking_uri("https://dagshub.com/Abhishek-B16/railway-defect-detection-mlops.mlflow")
    mlflow.set_experiment("0") # The default experiment in DagsHub
    
    # 2. Read the V1 results.csv
    csv_path = r"runs\detect\runs\detect\train\results.csv"
    if not os.path.exists(csv_path):
        print("Could not find V1 results.csv!")
        return
        
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names just in case
    df.columns = [c.strip() for c in df.columns]
    
    print("Uploading V1 metrics to DagsHub so you can compare them...")
    
    # 3. Create a new run for V1
    with mlflow.start_run(run_name="V1_Baseline_Run") as run:
        # Log parameters
        mlflow.log_param("model", "yolov8n.pt")
        mlflow.log_param("dataset", "V1_original")
        
        # Log each epoch's metrics
        for index, row in df.iterrows():
            epoch = int(row["epoch"])
            
            # Map YOLOv8 CSV columns to standard MLflow metrics
            metrics = {
                "train/box_loss": row["train/box_loss"],
                "train/cls_loss": row["train/cls_loss"],
                "metrics/precision(B)": row["metrics/precision(B)"],
                "metrics/recall(B)": row["metrics/recall(B)"],
                "metrics/mAP50(B)": row["metrics/mAP50(B)"],
                "metrics/mAP50-95(B)": row["metrics/mAP50-95(B)"],
                "val/box_loss": row["val/box_loss"],
                "val/cls_loss": row["val/cls_loss"]
            }
            mlflow.log_metrics(metrics, step=epoch)
            
        print("✅ SUCCESS! V1 Metrics have been uploaded to DagsHub.")
        print("Go back to your DagsHub Experiments tab and click 'Compare'!")

if __name__ == "__main__":
    upload_v1_metrics()
