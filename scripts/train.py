import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import yaml
import glob
import subprocess
import mlflow
from ultralytics import YOLO

def get_dvc_hash():
    try:
        with open("dataset_clean.dvc", "r") as f:
            for line in f:
                if "md5:" in line:
                    return line.split(":")[1].strip()
        return "unknown"
    except:
        return "unknown"

def count_images(split):
    img_dir = os.path.join("dataset_clean", split, "images")
    if not os.path.exists(img_dir): return 0
    return len(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.jpeg")) + glob.glob(os.path.join(img_dir, "*.png")))

def main():
    config_path = "configs/train.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Dataset stats
    train_imgs = count_images("train")
    val_imgs = count_images("valid")
    test_imgs = count_images("test")
    
    with open("dataset_clean/data.yaml", "r") as f:
        data_yaml = yaml.safe_load(f)
    num_classes = data_yaml.get("nc", 0)

    # MLflow Setup
    mlflow.set_experiment("RailGuard")
    
    with mlflow.start_run() as run:
        print(f"Started MLflow run: {run.info.run_id}")
        
        # Log custom parameters requested
        mlflow.log_params({
            "model_type": config["model"],
            "dataset_version_hash": get_dvc_hash(),
            "num_training_images": train_imgs,
            "num_validation_images": val_imgs,
            "num_test_images": test_imgs,
            "num_classes": num_classes,
            "epochs": config["epochs"],
            "imgsz": config["imgsz"],
            "batch": config["batch"],
            "optimizer": config["optimizer"],
            "learning_rate": config["lr0"],
            "seed": config["seed"]
        })

        # Load YOLO model
        model = YOLO(config["model"])
        
        # Train
        results = model.train(
            data=config["data"],
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            lr0=config["lr0"],
            optimizer=config["optimizer"],
            seed=config["seed"],
            device=config.get("device", ""),
            project=config["project"],
            name=config["name"],
            exist_ok=config["exist_ok"]
        )

        # Evaluate on Test dataset
        print("\n--- Evaluating on Test Dataset ---")
        test_results = model.val(data=config["data"], split='test')
        
        # Log custom test metrics (mAP50, mAP50-95, precision, recall)
        mlflow.log_metrics({
            "test_mAP50": test_results.box.map50,
            "test_mAP50-95": test_results.box.map,
            "test_precision": test_results.box.p.mean() if len(test_results.box.p) > 0 else 0.0,
            "test_recall": test_results.box.r.mean() if len(test_results.box.r) > 0 else 0.0
        })

        # Explicitly log artifacts from the YOLO output directory
        output_dir = os.path.join(config["project"], config["name"])
        
        if os.path.exists(output_dir):
            print(f"Logging artifacts from {output_dir}...")
            # Ultralytics natively logs to MLflow, but we ensure our custom run captures everything
            mlflow.log_artifacts(output_dir, artifact_path="yolo_outputs")
            
        print("Training complete. MLflow run logged successfully.")

if __name__ == "__main__":
    main()
