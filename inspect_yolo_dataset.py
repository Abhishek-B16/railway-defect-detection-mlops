import os
import glob
from PIL import Image
import hashlib
import yaml

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def inspect_yolo_dataset(dataset_path):
    print("Starting YOLO dataset inspection...")
    
    report = {
        "num_train_images": 0,
        "num_valid_images": 0,
        "num_test_images": 0,
        "num_label_files": 0,
        "image_label_matching": True,
        "classes": [],
        "num_classes": 0,
        "annotations_per_class": {},
        "bounding_boxes_exist": False,
        "annotation_format": "YOLO txt",
        "yolo_format_correct": True,
        "coordinates_normalized": True,
        "missing_labels": 0,
        "empty_labels": 0,
        "corrupted_images": 0,
        "invalid_class_ids": 0,
        "invalid_bounding_boxes": 0,
        "duplicate_images": 0,
        "image_dimensions": set(),
        "is_valid_for_yolo": True,
        "data_yaml_paths": {}
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
            for cls_name in report["classes"]:
                report["annotations_per_class"][cls_name] = 0
            
            report["data_yaml_paths"] = {
                "train": data.get("train", ""),
                "val": data.get("val", ""),
                "test": data.get("test", "")
            }
    else:
        print("data.yaml not found!")
        report["is_valid_for_yolo"] = False

    hashes = {}
    
    splits = ["train", "valid", "test"]
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
            
        images_path = os.path.join(split_path, "images")
        labels_path = os.path.join(split_path, "labels")
        
        # fallback to root split dir if images/labels subdirs don't exist
        if not os.path.exists(images_path): images_path = split_path
        if not os.path.exists(labels_path): labels_path = split_path

        img_files = glob.glob(os.path.join(images_path, "*.jpg")) + \
                    glob.glob(os.path.join(images_path, "*.jpeg")) + \
                    glob.glob(os.path.join(images_path, "*.png"))
        
        if split == "train": report["num_train_images"] = len(img_files)
        elif split == "valid": report["num_valid_images"] = len(img_files)
        elif split == "test": report["num_test_images"] = len(img_files)
        
        for img_path in img_files:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    report["image_dimensions"].add(img.size)
            except Exception as e:
                report["corrupted_images"] += 1
                report["is_valid_for_yolo"] = False
                continue
                
            file_hash = get_file_hash(img_path)
            if file_hash in hashes:
                report["duplicate_images"] += 1
            else:
                hashes[file_hash] = True
                
            # Check corresponding label
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_path, base_name + ".txt")
            
            if not os.path.exists(label_path):
                report["missing_labels"] += 1
                report["image_label_matching"] = False
                continue
                
            report["num_label_files"] += 1
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    report["empty_labels"] += 1
                    continue
                    
                report["bounding_boxes_exist"] = True
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        report["yolo_format_correct"] = False
                        report["is_valid_for_yolo"] = False
                        continue
                    
                    try:
                        cls_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        if cls_id < 0 or cls_id >= report["num_classes"]:
                            report["invalid_class_ids"] += 1
                            report["is_valid_for_yolo"] = False
                        else:
                            cls_name = report["classes"][cls_id]
                            report["annotations_per_class"][cls_name] = report["annotations_per_class"].get(cls_name, 0) + 1
                            
                        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 
                                0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
                            report["coordinates_normalized"] = False
                            report["invalid_bounding_boxes"] += 1
                            report["is_valid_for_yolo"] = False
                            
                    except ValueError:
                        report["yolo_format_correct"] = False
                        report["is_valid_for_yolo"] = False

    if report["missing_labels"] > 0 or report["corrupted_images"] > 0 or not report["yolo_format_correct"] or not report["coordinates_normalized"]:
         report["is_valid_for_yolo"] = False

    import json
    with open("dataset_inspection_results.json", "w") as f:
        # Convert sets to list for JSON serialization
        report["image_dimensions"] = list(report["image_dimensions"])
        json.dump(report, f, indent=4)
        
    print("Inspection complete.")

if __name__ == '__main__':
    dataset_dir = r"c:\Users\ABHISHEK\ML project\railway-defect-mlops\railway defect dataset.v9i.yolov8"
    inspect_yolo_dataset(dataset_dir)
