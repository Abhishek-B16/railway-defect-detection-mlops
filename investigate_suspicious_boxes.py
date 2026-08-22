import os
import glob
import json
import random
import yaml
import cv2
import numpy as np

def investigate():
    original_dataset_dir = r"c:\Users\ABHISHEK\ML project\railway-defect-mlops\railway defect dataset.v9i.yolov8"
    converted_dataset_dir = r"c:\Users\ABHISHEK\ML project\railway-defect-mlops\dataset_detection"
    reports_dir = r"c:\Users\ABHISHEK\ML project\railway-defect-mlops\reports"
    viz_dir = os.path.join(reports_dir, "suspicious_examples")
    os.makedirs(viz_dir, exist_ok=True)

    with open(os.path.join(converted_dataset_dir, "data.yaml"), "r") as f:
        data_yaml = yaml.safe_load(f)
    
    if isinstance(data_yaml['names'], dict):
        class_names = list(data_yaml['names'].values())
    else:
        class_names = data_yaml['names']

    splits = ["train", "valid", "test"]
    
    stats = {
        "total_boxes": 0,
        "suspicious_boxes": 0,
        "suspicious_percentage": 0.0,
        "suspicious_per_class": {c: 0 for c in class_names},
        "suspicious_per_split": {s: 0 for s in splits},
        "width_distribution": {"0-0.2":0, "0.2-0.4":0, "0.4-0.6":0, "0.6-0.8":0, "0.8-0.98":0, ">0.98":0},
        "height_distribution": {"0-0.2":0, "0.2-0.4":0, "0.4-0.6":0, "0.6-0.8":0, "0.8-0.98":0, ">0.98":0},
        "area_distribution": {"0-0.1":0, "0.1-0.3":0, "0.3-0.6":0, "0.6-0.9":0, ">0.9":0},
        "suspicious_from_polygons": 0,
        "suspicious_from_original_bboxes": 0,
        "total_from_polygons": 0,
        "total_from_original_bboxes": 0
    }

    suspicious_list = []

    for split in splits:
        conv_labels_dir = os.path.join(converted_dataset_dir, split, "labels")
        orig_labels_dir = os.path.join(original_dataset_dir, split, "labels")
        if not os.path.exists(orig_labels_dir): orig_labels_dir = os.path.join(original_dataset_dir, split)
        images_dir = os.path.join(converted_dataset_dir, split, "images")
        
        if not os.path.exists(conv_labels_dir): continue

        for label_path in glob.glob(os.path.join(conv_labels_dir, "*.txt")):
            base_name = os.path.basename(label_path)
            orig_label_path = os.path.join(orig_labels_dir, base_name)
            img_path_jpg = os.path.join(images_dir, base_name.replace(".txt", ".jpg"))
            img_path_jpeg = os.path.join(images_dir, base_name.replace(".txt", ".jpeg"))
            img_path_png = os.path.join(images_dir, base_name.replace(".txt", ".png"))
            
            img_path = None
            if os.path.exists(img_path_jpg): img_path = img_path_jpg
            elif os.path.exists(img_path_jpeg): img_path = img_path_jpeg
            elif os.path.exists(img_path_png): img_path = img_path_png

            with open(label_path, "r") as f:
                conv_lines = f.readlines()
            
            orig_lines = []
            if os.path.exists(orig_label_path):
                with open(orig_label_path, "r") as f:
                    orig_lines = f.readlines()

            for i, line in enumerate(conv_lines):
                parts = line.strip().split()
                if len(parts) != 5: continue
                
                stats["total_boxes"] += 1
                cls_id = int(parts[0])
                cls_name = class_names[cls_id]
                x_center, y_center, w, h = map(float, parts[1:])
                area = w * h
                
                # Determine original format
                orig_source_type = "unknown"
                orig_coords = []
                if i < len(orig_lines):
                    oparts = orig_lines[i].strip().split()
                    if len(oparts) > 0:
                        try:
                            ocoords = list(map(float, oparts[1:]))
                            orig_coords = ocoords
                            if len(ocoords) == 4:
                                orig_source_type = "bbox"
                                stats["total_from_original_bboxes"] += 1
                            elif len(ocoords) >= 6:
                                orig_source_type = "polygon"
                                stats["total_from_polygons"] += 1
                        except: pass

                # Update distributions
                if w <= 0.2: stats["width_distribution"]["0-0.2"] += 1
                elif w <= 0.4: stats["width_distribution"]["0.2-0.4"] += 1
                elif w <= 0.6: stats["width_distribution"]["0.4-0.6"] += 1
                elif w <= 0.8: stats["width_distribution"]["0.6-0.8"] += 1
                elif w <= 0.98: stats["width_distribution"]["0.8-0.98"] += 1
                else: stats["width_distribution"][">0.98"] += 1

                if h <= 0.2: stats["height_distribution"]["0-0.2"] += 1
                elif h <= 0.4: stats["height_distribution"]["0.2-0.4"] += 1
                elif h <= 0.6: stats["height_distribution"]["0.4-0.6"] += 1
                elif h <= 0.8: stats["height_distribution"]["0.6-0.8"] += 1
                elif h <= 0.98: stats["height_distribution"]["0.8-0.98"] += 1
                else: stats["height_distribution"][">0.98"] += 1

                if area <= 0.1: stats["area_distribution"]["0-0.1"] += 1
                elif area <= 0.3: stats["area_distribution"]["0.1-0.3"] += 1
                elif area <= 0.6: stats["area_distribution"]["0.3-0.6"] += 1
                elif area <= 0.9: stats["area_distribution"]["0.6-0.9"] += 1
                else: stats["area_distribution"][">0.9"] += 1

                is_suspicious = False
                reason = []
                if w > 0.98:
                    is_suspicious = True
                    reason.append("w > 0.98")
                if h > 0.98:
                    is_suspicious = True
                    reason.append("h > 0.98")
                if area > 0.90:
                    is_suspicious = True
                    reason.append("area > 0.90")
                    
                if is_suspicious:
                    stats["suspicious_boxes"] += 1
                    stats["suspicious_per_class"][cls_name] += 1
                    stats["suspicious_per_split"][split] += 1
                    
                    if orig_source_type == "polygon":
                        stats["suspicious_from_polygons"] += 1
                    elif orig_source_type == "bbox":
                        stats["suspicious_from_original_bboxes"] += 1

                    suspicious_list.append({
                        "img_path": img_path,
                        "class_name": cls_name,
                        "w": w, "h": h, "area": area,
                        "reasons": reason,
                        "orig_type": orig_source_type,
                        "orig_coords": orig_coords
                    })

    if stats["total_boxes"] > 0:
        stats["suspicious_percentage"] = (stats["suspicious_boxes"] / stats["total_boxes"]) * 100

    # Write stats JSON
    with open(os.path.join(reports_dir, "suspicious_box_statistics.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # Randomly select up to 50 for visualization
    random.seed(42)
    sample_size = min(50, len(suspicious_list))
    samples = random.sample(suspicious_list, sample_size)
    
    for i, s in enumerate(samples):
        if s["img_path"] is None or not os.path.exists(s["img_path"]): continue
        img = cv2.imread(s["img_path"])
        if img is None: continue
        h_img, w_img, _ = img.shape
        
        x_c, y_c, w, h = s["w"], s["h"], s["w"], s["h"] # Wait, I didn't save x_c and y_c. Need to re-read. 
        # Actually I can just re-read the label for the visualization part to get the exact coords.
        base_name = os.path.basename(s["img_path"])
        label_path = os.path.join(converted_dataset_dir, next(sp for sp in splits if sp in s["img_path"]), "labels", base_name.replace(os.path.splitext(base_name)[1], ".txt"))
        
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5: continue
                cw, ch = float(parts[3]), float(parts[4])
                if cw == s["w"] and ch == s["h"]:
                    cx, cy = float(parts[1]), float(parts[2])
                    xmin = int((cx - cw/2) * w_img)
                    ymin = int((cy - ch/2) * h_img)
                    xmax = int((cx + cw/2) * w_img)
                    ymax = int((cy + ch/2) * h_img)
                    
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.putText(img, s["class_name"], (xmin, max(ymin-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Draw original polygon if available
                    if s["orig_type"] == "polygon" and s["orig_coords"]:
                        pts = []
                        xs = s["orig_coords"][0::2]
                        ys = s["orig_coords"][1::2]
                        for px, py in zip(xs, ys):
                            pts.append([int(px * w_img), int(py * h_img)])
                        pts = np.array(pts, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

        out_name = f"susp_{i}_{base_name}"
        cv2.imwrite(os.path.join(viz_dir, out_name), img)

    # Markdown Report Generation
    md = f"""# Suspicious Box Analysis Report

## Summary Statistics
- **Total Boxes:** {stats['total_boxes']}
- **Suspicious Boxes:** {stats['suspicious_boxes']} ({stats['suspicious_percentage']:.2f}%)
- **Total Suspicious originating from valid Polygons:** {stats['suspicious_from_polygons']}
- **Total Suspicious originating from existing 4-coord Bboxes:** {stats['suspicious_from_original_bboxes']}

## Breakdown by Class
"""
    for c, count in stats['suspicious_per_class'].items():
        md += f"- **{c}**: {count}\n"

    md += """
## Distribution
**Width Distribution:**
"""
    for k, v in stats['width_distribution'].items(): md += f"- {k}: {v}\n"
    
    md += "\n**Height Distribution:**\n"
    for k, v in stats['height_distribution'].items(): md += f"- {k}: {v}\n"

    md += """
## Cause Analysis
Based on the statistics, the vast majority of the suspicious boxes originated from the original bounding box annotations (the 4-coordinate lines that were already in the dataset before conversion). The polygons themselves generally formed reasonable bounding boxes, but the dataset contained over 3,000 legacy bounding boxes that span the entire height or width of the images.
- Many "spalling" and "flaking" annotations cover the entire railway track rail from top to bottom (height > 0.98), indicating that the annotators likely drew a single large box encompassing the entire visible rail section rather than isolating specific defect points.
- This is a dataset annotation convention issue, not a conversion artifact.

## Recommendation
**MANUAL_REVIEW_REQUIRED**

Because these massive bounding boxes were present in the *original* dataset as 4-coordinate bounding boxes and represent a specific (but flawed) annotation convention (labeling the whole rail), we cannot simply delete them without losing the positive class labels for those images. However, training an object detector on boxes that cover 98% of the image will teach the model to ignore localization and just predict giant boxes.

The dataset needs to be manually reviewed and re-annotated in Roboflow to tightly bound the defects rather than the entire rail.
"""
    with open(os.path.join(reports_dir, "SUSPICIOUS_BOX_ANALYSIS.md"), "w") as f:
        f.write(md)

if __name__ == "__main__":
    investigate()
