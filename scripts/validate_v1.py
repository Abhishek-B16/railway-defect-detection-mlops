from ultralytics import YOLO
import os
import json

def validate_baseline():
    print("=== Phase 2: Baseline Validation (V1) ===")
    
    model_path = r"C:\Users\ABHISHEK\ML project\railway-defect-mlops\mlruns\1\80842181f7e345bb85b961dc2fdcfd3c\artifacts\model\best.pt"
    data_path = r"C:\Users\ABHISHEK\ML project\railway-defect-mlops\dataset_clean\data.yaml"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print("Loading V1 model...")
    model = YOLO(model_path)
    
    print(f"Running validation on {data_path}...")
    # Run validation
    metrics = model.val(data=data_path, split='val', device='cpu') # Use cpu for validation if gpu not strictly needed or '0' for gpu
    
    results = {
        'map50': metrics.box.map50,
        'map50_95': metrics.box.map,
        'precision': metrics.box.mp,
        'recall': metrics.box.mr,
        'fitness': metrics.box.fitness,
        'per_class': {}
    }
    
    # Save per-class metrics
    class_indices = metrics.box.ap_class_index
    for i, c in enumerate(class_indices):
        class_name = model.names[c]
        results['per_class'][class_name] = {
            'map50': metrics.box.ap50[i],
            'map50_95': metrics.box.ap[i],
            'precision': metrics.box.p[i],
            'recall': metrics.box.r[i]
        }
        
    # Write to a JSON file to parse easily later
    with open('v1_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nV1 Baseline Metrics:")
    print(f"mAP50: {results['map50']:.4f}")
    print(f"mAP50-95: {results['map50_95']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    
    print("\nPer-class mAP50:")
    for cls, cm in results['per_class'].items():
        print(f"  {cls}: {cm['map50']:.4f}")

if __name__ == '__main__':
    validate_baseline()
