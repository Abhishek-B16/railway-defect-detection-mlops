import os
import glob
import json
import shutil
import csv
import random
import yaml
import cv2
from PIL import Image

def convert_dataset():
    source_dir = r"c:\Users\ABHISHEK\ML project\railway-defect-mlops\railway defect dataset.v9i.yolov8"
    dest_dir = r"c:\Users\ABHISHEK\ML project\railway-defect-mlops\dataset_detection"
    reports_dir = r"c:\Users\ABHISHEK\ML project\railway-defect-mlops\reports"
    viz_dir = os.path.join(dest_dir, "visualizations")

    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    splits = ["train", "valid", "test"]
    for split in splits:
        os.makedirs(os.path.join(dest_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, split, "labels"), exist_ok=True)

    with open(os.path.join(source_dir, "data.yaml"), "r") as f:
        data_yaml = yaml.safe_load(f)

    # Prepare stats
    stats = {
        "original_image_count": 0,
        "original_annotation_count": 0,
        "valid_polygon_count": 0,
        "converted_bbox_count": 0,
        "excluded_annotation_count": 0,
        "annotations_per_class_before": {cls: 0 for cls in data_yaml['names']},
        "annotations_per_class_after": {cls: 0 for cls in data_yaml['names']},
        "train_count": 0,
        "valid_count": 0,
        "test_count": 0,
        "images_affected_by_excluded": 0
    }

    if isinstance(data_yaml['names'], dict):
        class_names = list(data_yaml['names'].values())
    else:
        class_names = data_yaml['names']

    excluded_annotations = []
    converted_samples = [] # (img_path, label_path) for visualization
    affected_images_set = set()

    for split in splits:
        source_split = os.path.join(source_dir, split)
        if not os.path.exists(source_split):
            continue
            
        images_dir = os.path.join(source_split, "images")
        labels_dir = os.path.join(source_split, "labels")

        if not os.path.exists(images_dir): images_dir = source_split
        if not os.path.exists(labels_dir): labels_dir = source_split

        img_files = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                    glob.glob(os.path.join(images_dir, "*.jpeg")) + \
                    glob.glob(os.path.join(images_dir, "*.png"))

        for img_path in img_files:
            stats["original_image_count"] += 1
            if split == "train": stats["train_count"] += 1
            elif split == "valid": stats["valid_count"] += 1
            elif split == "test": stats["test_count"] += 1

            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(labels_dir, base_name + ".txt")
            
            dest_img_path = os.path.join(dest_dir, split, "images", img_name)
            dest_label_path = os.path.join(dest_dir, split, "labels", base_name + ".txt")

            shutil.copy(img_path, dest_img_path)

            if not os.path.exists(label_path):
                open(dest_label_path, 'w').close()
                continue

            converted_lines = []
            has_excluded = False

            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if not parts:
                        continue

                    stats["original_annotation_count"] += 1
                    try:
                        cls_id = int(parts[0])
                        cls_name = class_names[cls_id]
                        stats["annotations_per_class_before"][cls_name] += 1
                    except (ValueError, IndexError):
                        excluded_annotations.append({
                            "image": img_name, "line": line_idx, "reason": "Invalid class ID"
                        })
                        stats["excluded_annotation_count"] += 1
                        has_excluded = True
                        continue

                    try:
                        coords = list(map(float, parts[1:]))
                    except ValueError:
                        excluded_annotations.append({
                            "image": img_name, "line": line_idx, "reason": "Non-numeric coordinates"
                        })
                        stats["excluded_annotation_count"] += 1
                        has_excluded = True
                        continue

                    # Determine if valid polygon or bounding box
                    if len(coords) < 6:
                        # Less than 3 coordinate pairs. Is it a standard bounding box?
                        if len(coords) == 4:
                            # Safely interpreted as bounding box (x_center, y_center, w, h)
                            stats["converted_bbox_count"] += 1
                            stats["annotations_per_class_after"][cls_name] += 1
                            converted_lines.append(f"{cls_id} {coords[0]} {coords[1]} {coords[2]} {coords[3]}\n")
                        else:
                            excluded_annotations.append({
                                "image": img_name, "line": line_idx, "reason": f"Insufficient coordinates ({len(coords)})"
                            })
                            stats["excluded_annotation_count"] += 1
                            has_excluded = True
                    else:
                        if len(coords) % 2 != 0:
                            excluded_annotations.append({
                                "image": img_name, "line": line_idx, "reason": "Odd number of coordinates"
                            })
                            stats["excluded_annotation_count"] += 1
                            has_excluded = True
                            continue

                        # Valid polygon -> bounding box
                        stats["valid_polygon_count"] += 1
                        
                        xs = coords[0::2]
                        ys = coords[1::2]
                        
                        xmin, xmax = min(xs), max(xs)
                        ymin, ymax = min(ys), max(ys)
                        
                        # Clip to [0, 1]
                        xmin = max(0.0, min(1.0, xmin))
                        xmax = max(0.0, min(1.0, xmax))
                        ymin = max(0.0, min(1.0, ymin))
                        ymax = max(0.0, min(1.0, ymax))

                        width = xmax - xmin
                        height = ymax - ymin
                        x_center = xmin + width / 2.0
                        y_center = ymin + height / 2.0

                        if width > 0 and height > 0:
                            stats["converted_bbox_count"] += 1
                            stats["annotations_per_class_after"][cls_name] += 1
                            converted_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        else:
                            excluded_annotations.append({
                                "image": img_name, "line": line_idx, "reason": "Degenerate bounding box (w=0 or h=0)"
                            })
                            stats["excluded_annotation_count"] += 1
                            has_excluded = True
                            
            if has_excluded:
                affected_images_set.add(img_name)

            if converted_lines:
                converted_samples.append((dest_img_path, dest_label_path))

            with open(dest_label_path, 'w') as f:
                f.writelines(converted_lines)

    stats["images_affected_by_excluded"] = len(affected_images_set)

    # New data.yaml
    new_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names
    }
    with open(os.path.join(dest_dir, "data.yaml"), "w") as f:
        yaml.dump(new_yaml, f)

    # Write Excluded CSV
    with open(os.path.join(reports_dir, "excluded_annotations.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "line", "reason"])
        writer.writeheader()
        writer.writerows(excluded_annotations)

    # Write Stats JSON
    with open(os.path.join(reports_dir, "conversion_statistics.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # Write Conversion Report
    report_content = f"""# Dataset Conversion Report

## Conversion Summary
- **Original image count:** {stats['original_image_count']}
- **Original annotation count:** {stats['original_annotation_count']}
- **Valid polygon count (converted):** {stats['valid_polygon_count']}
- **Converted bounding-box count:** {stats['converted_bbox_count']}
- **Excluded annotation count:** {stats['excluded_annotation_count']}
- **Images affected by excluded annotations:** {stats['images_affected_by_excluded']}

## Data Splits
- **Train count:** {stats['train_count']}
- **Valid count:** {stats['valid_count']}
- **Test count:** {stats['test_count']}

## Excluded Annotation Reasons
Check `excluded_annotations.csv` for detailed rows. Common reasons: Insufficient coordinates, non-numeric values, or degenerate boxes.

## Class Distribution Before vs After

| Class Name | Annotations Before | Annotations After |
|---|---|---|
"""
    for cls in class_names:
        report_content += f"| {cls} | {stats['annotations_per_class_before'][cls]} | {stats['annotations_per_class_after'][cls]} |\n"

    with open(os.path.join(reports_dir, "conversion_report.md"), "w") as f:
        f.write(report_content)

    print("Conversion complete. Proceeding to Strict Validation.")

    # Validation Phase
    validation_passed = True
    for split in splits:
        labels_dir = os.path.join(dest_dir, split, "labels")
        if not os.path.exists(labels_dir): continue
        for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
            with open(label_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Validation Error: {label_file} has a line with {len(parts)} parts.")
                        validation_passed = False
                    try:
                        cls_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        if cls_id < 0 or cls_id >= len(class_names):
                            print(f"Validation Error: {label_file} has invalid class ID {cls_id}.")
                            validation_passed = False
                        if not all(0 <= v <= 1 for v in [x, y, w, h]):
                            print(f"Validation Error: {label_file} coords not normalized.")
                            validation_passed = False
                        if w <= 0 or h <= 0:
                            print(f"Validation Error: {label_file} box width/height <= 0.")
                            validation_passed = False
                    except Exception as e:
                        print(f"Validation Error: {label_file} parsing failed. {str(e)}")
                        validation_passed = False

    if validation_passed:
        print("STRICT VALIDATION PASSED.")
    else:
        print("STRICT VALIDATION FAILED.")

    # Visualizations
    random.seed(42)
    sample_size = min(20, len(converted_samples))
    visual_samples = random.sample(converted_samples, sample_size)
    
    for i, (img_path, label_path) in enumerate(visual_samples):
        img = cv2.imread(img_path)
        if img is None: continue
        h, w_img, _ = img.shape
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5: continue
                cls_id = int(parts[0])
                x_center, y_center, bbox_w, bbox_h = map(float, parts[1:])
                
                xmin = int((x_center - bbox_w/2) * w_img)
                ymin = int((y_center - bbox_h/2) * h)
                xmax = int((x_center + bbox_w/2) * w_img)
                ymax = int((y_center + bbox_h/2) * h)
                
                color = ((cls_id * 50) % 255, (cls_id * 100) % 255, (cls_id * 150 + 100) % 255)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                cls_name = class_names[cls_id]
                cv2.putText(img, cls_name, (xmin, max(ymin-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out_path = os.path.join(viz_dir, f"viz_{i}_{os.path.basename(img_path)}")
        cv2.imwrite(out_path, img)

    print(f"Generated {sample_size} visualizations in {viz_dir}")

if __name__ == '__main__':
    convert_dataset()
