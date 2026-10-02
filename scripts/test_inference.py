from ultralytics import YOLO
import cv2

# Load the V3 model
model = YOLO('yolov8n.pt')  # This is the V3 model we copied to the root

# Test an image from the validation set
# We'll use the one the user just tried if it's in the dataset
img_path = r"dataset_clean\valid\images\128_jpg.rf.0cba14a74173cd77c467134ff5c85f56.jpg"

print(f"Testing inference on {img_path}...")
results = model(img_path, conf=0.01, imgsz=640)

if len(results) > 0:
    boxes = results[0].boxes
    print(f"Detected {len(boxes)} objects:")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        print(f"- {class_name}: {conf:.2f}")
else:
    print("Detected 0 objects.")
