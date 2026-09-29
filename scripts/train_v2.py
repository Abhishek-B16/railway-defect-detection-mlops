from ultralytics import YOLO
import os

def train_v2():
    print("=== Phase 4: Training Improved Model (V2) ===")
    model = YOLO('yolov8n.pt') 
    
    # Train
    results = model.train(
        data=r'C:\Users\ABHISHEK\ML project\railway-defect-mlops\dataset_clean\data.yaml',
        epochs=30, # A bit shorter to ensure it finishes quickly for this experiment
        imgsz=640,
        patience=10,
        project='runs/train',
        name='v2_experiment',
        device='0' if __import__('torch').cuda.is_available() else 'cpu'
    )
    
if __name__ == '__main__':
    train_v2()
