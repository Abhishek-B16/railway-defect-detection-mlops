import os
import pandas as pd
import mlflow
import getpass

def upload_metrics():
    # 1. Setup MLflow to point to DagsHub
    print("To authorize DagsHub, please paste your Access Token below:")
    token = getpass.getpass("Token (it will be invisible as you type): ")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Abhishek-B16"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token.strip()
    
    mlflow.set_tracking_uri("https://dagshub.com/Abhishek-B16/railway-defect-detection-mlops.mlflow")
    
    # Force it to log to the exact experiment you are looking at right now
    mlflow.set_experiment("Railway-Defect-Detection") 
    
    # 2. Upload V3 Metrics
    v3_csv = r"runs\detect\runs\train\v3_experiment-2\results.csv"
    if os.path.exists(v3_csv):
        print("Uploading V3 metrics...")
        df3 = pd.read_csv(v3_csv)
        df3.columns = [c.strip() for c in df3.columns]
        df3 = df3.fillna(0) # Fix NaN values
        with mlflow.start_run(run_name="V3_YOLO_Nano") as run:
            mlflow.log_param("model", "yolov8n.pt")
            mlflow.log_param("dataset", "Imbalanced (54 cracks)")
            for index, row in df3.iterrows():
                metrics = {
                    "train_box_loss": float(row.get("train/box_loss", 0)),
                    "metrics_mAP50": float(row.get("metrics/mAP50(B)", 0)),
                    "metrics_mAP50_95": float(row.get("metrics/mAP50-95(B)", 0))
                }
                mlflow.log_metrics(metrics, step=int(row["epoch"]))
    
    # 3. Upload V4 Metrics
    v4_csv = r"runs\detect\runs\train\v4_experiment\results.csv"
    if os.path.exists(v4_csv):
        print("Uploading V4 metrics...")
        df4 = pd.read_csv(v4_csv)
        df4.columns = [c.strip() for c in df4.columns]
        df4 = df4.fillna(0) # Fix NaN values
        with mlflow.start_run(run_name="V4_YOLO_Medium") as run:
            mlflow.log_param("model", "yolov8m.pt")
            mlflow.log_param("dataset", "Balanced (718 cracks added)")
            for index, row in df4.iterrows():
                metrics = {
                    "train_box_loss": float(row.get("train/box_loss", 0)),
                    "metrics_mAP50": float(row.get("metrics/mAP50(B)", 0)),
                    "metrics_mAP50_95": float(row.get("metrics/mAP50-95(B)", 0))
                }
                mlflow.log_metrics(metrics, step=int(row["epoch"]))

    print("✅ SUCCESS! V3 and V4 Metrics are now in the Railway-Defect-Detection Experiment!")

if __name__ == "__main__":
    upload_metrics()
