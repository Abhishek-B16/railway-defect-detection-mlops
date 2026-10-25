from ultralytics import YOLO
import os

def train_v5():
    print("=== Phase 7: Training V5 (YOLOv8-Large + HD Single Block Cracks) ===")
    
    # 1. Setup MLflow Tracking
    import mlflow
    mlflow.set_tracking_uri("https://dagshub.com/Abhishek-B16/railway-defect-detection-mlops.mlflow")
    
    # Force it into the main experiment folder so it shows up on DagsHub!
    mlflow.set_experiment("Railway-Defect-Detection") 
    
    # 2. UPGRADE: Use the massive YOLOv8-Large architecture!
    model = YOLO('yolov8l.pt') 
    
    # 3. Train
    print("Starting Training on 6 Classes (Bolts, Flaking, Joints, Sheling, Spalling, and HD Cracks)")
    results = model.train(
        data=r'C:\Users\ABHISHEK\ML project\railway-defect-mlops\dataset_clean\data.yaml',
        epochs=30,       # Start with 30 epochs. If it hits 60%, we can push further!
        imgsz=640,
        batch=4,         # Large models take up lots of GPU RAM. Batch=4 prevents crashing.
        patience=10,     
        project='runs/train',
        name='v5_experiment',
        device='0' if __import__('torch').cuda.is_available() else 'cpu'
    )
    
if __name__ == '__main__':
    train_v5()
