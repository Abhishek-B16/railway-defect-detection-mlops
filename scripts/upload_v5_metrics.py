import os
import pandas as pd
import mlflow
import getpass

def upload_v5_metrics():
    print("To authorize DagsHub, please paste your Access Token below:")
    token = getpass.getpass("Token (it will be invisible as you type): ")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Abhishek-B16"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token.strip()
    
    mlflow.set_tracking_uri("https://dagshub.com/Abhishek-B16/railway-defect-detection-mlops.mlflow")
    mlflow.set_experiment("Railway-Defect-Detection") 
    
    v5_csv = r"runs\detect\runs\train\v5_experiment-2\results.csv"
    if os.path.exists(v5_csv):
        print("Uploading V5 metrics...")
        df5 = pd.read_csv(v5_csv)
        df5.columns = [c.strip() for c in df5.columns]
        df5 = df5.fillna(0) # Fix NaN values
        with mlflow.start_run(run_name="V5_YOLO_Large_HD") as run:
            mlflow.log_param("model", "yolov8l.pt")
            mlflow.log_param("dataset", "HD Single Block (Cracks Only)")
            for index, row in df5.iterrows():
                metrics = {
                    "train_box_loss": float(row.get("train/box_loss", 0)),
                    "metrics_mAP50": float(row.get("metrics/mAP50(B)", 0)),
                    "metrics_mAP50_95": float(row.get("metrics/mAP50-95(B)", 0))
                }
                mlflow.log_metrics(metrics, step=int(row["epoch"]))

        print("✅ SUCCESS! V5 Metrics uploaded to DagsHub!")
    else:
        print(f"Could not find V5 csv at: {v5_csv}")

if __name__ == "__main__":
    upload_v5_metrics()
