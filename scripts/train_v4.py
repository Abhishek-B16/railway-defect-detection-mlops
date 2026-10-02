from ultralytics import YOLO
import os

def train_v4():
    print("=== Phase 6: Training V4 (Balanced Cracks) ===")
    
    # Force MLflow to log to DagsHub
    import mlflow
    mlflow.set_tracking_uri("https://dagshub.com/Abhishek-B16/railway-defect-detection-mlops.mlflow")
    
    # UPGRADE: Using 'Medium' model instead of 'Nano' for much higher accuracy
    model = YOLO('yolov8m.pt') 
    
    # Train
    results = model.train(
        data=r'C:\Users\ABHISHEK\ML project\railway-defect-mlops\dataset_clean\data.yaml',
        epochs=50,       # UPGRADE: Train for 50 epochs (balanced for time and accuracy)
        imgsz=640,
        batch=8,         # Set to 8 to ensure it fits in RTX 4050 memory
        patience=20,     # Wait 20 epochs before early stopping
        project='runs/train',
        name='v4_experiment',
        device='0' if __import__('torch').cuda.is_available() else 'cpu'
    )
    
if __name__ == '__main__':
    train_v4()
