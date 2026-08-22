# Training Guide: YOLO Object Detection

This document explains how to reproduce the baseline object detection training using YOLOv8 and MLflow.

## 1. Environment Requirements
Ensure your Python environment has the necessary packages:
```bash
pip install ultralytics mlflow
```

## 2. Configuration
The training pipeline is controlled entirely via `configs/train.yaml`.
You do not need to edit the training script itself to change hyperparameters.

```yaml
model: "yolov8n.pt"  # The base model checkpoint (n = nano, s = small, m = medium)
data: "c:/Users/ABHISHEK/ML project/railway-defect-mlops/dataset_clean/data.yaml"
epochs: 10
imgsz: 640
batch: 16
lr0: 0.01
optimizer: "auto"
seed: 42
project: "runs/detect"
name: "train"
```

## 3. Running the Training Pipeline
To launch the training pipeline with full MLflow tracking:
```bash
python scripts/train.py
```

### What happens during training?
1. The script reads the DVC hash from `dataset_clean.dvc` to link the model exactly to the data version.
2. It initializes an MLflow run under the experiment name `RailGuard`.
3. YOLOv8 begins training on the specified epochs.
4. At the end of training, the script evaluates the best model against the `test` split.
5. All metrics (mAP50, Precision, Recall), plots (Confusion matrix, Precision-Recall curve), and model weights are explicitly logged into MLflow.

## 4. Monitoring Experiments via MLflow
To view the results visually, start the MLflow UI:
```bash
mlflow ui
```
Then navigate your browser to `http://127.0.0.1:5000`. You will see the `RailGuard` experiment with the logged parameters, metrics, and generated artifacts.

## 5. Output Artifacts Location
If you prefer not to use the MLflow UI, all YOLO outputs are stored locally in:
`runs/detect/train/`

The best weights for inference are located at:
`runs/detect/train/weights/best.pt`
