import os
import glob
import yaml
import json

def inspect_segmentation_dataset(dataset_path):
    report = {
        "num_classes": 0,
        "classes": [],
        "images_train": 0,
        "images_valid": 0,
        "images_test": 0,
        "total_polygons": 0,
        "valid_polygons": 0,
        "invalid_polygons": 0,
        "polygons_min_3_pairs": True,
        "coordinates_normalized": True,
        "image_label_matching": True,
        "missing_labels": 0,
        "empty_labels": 0,
        "total_images": 0
    }

    # Inspect data.yaml
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            report["classes"] = data.get("names", [])
            if isinstance(report["classes"], dict):
                 report["classes"] = list(report["classes"].values())
            report["num_classes"] = data.get("nc", len(report["classes"]))

    splits = ["train", "valid", "test"]
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
            
        images_path = os.path.join(split_path, "images")
        labels_path = os.path.join(split_path, "labels")

        if not os.path.exists(images_path): images_path = split_path
        if not os.path.exists(labels_path): labels_path = split_path

        img_files = glob.glob(os.path.join(images_path, "*.jpg")) + \
                    glob.glob(os.path.join(images_path, "*.jpeg")) + \
                    glob.glob(os.path.join(images_path, "*.png"))
        
        num_images = len(img_files)
        report["total_images"] += num_images
        
        if split == "train": report["images_train"] = num_images
        elif split == "valid": report["images_valid"] = num_images
        elif split == "test": report["images_test"] = num_images
        
        for img_path in img_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_path, base_name + ".txt")
            
            if not os.path.exists(label_path):
                report["missing_labels"] += 1
                report["image_label_matching"] = False
                continue
                
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    report["empty_labels"] += 1
                    continue
                    
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    report["total_polygons"] += 1
                    
                    try:
                        coords = list(map(float, parts[1:]))
                        # Check for min 3 pairs (6 coordinates)
                        if len(coords) < 6 or len(coords) % 2 != 0:
                            report["invalid_polygons"] += 1
                            report["polygons_min_3_pairs"] = False
                            continue
                            
                        normalized = all(0.0 <= c <= 1.0 for c in coords)
                        if not normalized:
                            report["coordinates_normalized"] = False
                            report["invalid_polygons"] += 1
                        else:
                            report["valid_polygons"] += 1
                    except ValueError:
                        report["invalid_polygons"] += 1

    report["avg_annotations_per_image"] = report["total_polygons"] / report["total_images"] if report["total_images"] > 0 else 0

    with open("segmentation_inspection_results.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("Inspection complete.")

if __name__ == '__main__':
    dataset_dir = r"c:\Users\ABHISHEK\ML project\railway-defect-mlops\railway defect dataset.v9i.yolov8"
    inspect_segmentation_dataset(dataset_dir)
