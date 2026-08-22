import os
import glob
import shutil
import yaml

def clean_dataset():
    src_dir = r"c:\Users\ABHISHEK\ML project\railway-defect-mlops\dataset_detection"
    dest_dir = r"c:\Users\ABHISHEK\ML project\railway-defect-mlops\dataset_clean"
    
    os.makedirs(dest_dir, exist_ok=True)
    
    with open(os.path.join(src_dir, "data.yaml"), "r") as f:
        data_yaml = yaml.safe_load(f)
        
    stats = {
        "images_kept": 0,
        "images_discarded": 0,
        "annotations_kept": 0,
        "annotations_discarded": 0
    }

    splits = ["train", "valid", "test"]
    for split in splits:
        os.makedirs(os.path.join(dest_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, split, "labels"), exist_ok=True)
        
        src_images_dir = os.path.join(src_dir, split, "images")
        src_labels_dir = os.path.join(src_dir, split, "labels")
        
        dest_images_dir = os.path.join(dest_dir, split, "images")
        dest_labels_dir = os.path.join(dest_dir, split, "labels")

        if not os.path.exists(src_images_dir): continue

        img_files = glob.glob(os.path.join(src_images_dir, "*.jpg")) + \
                    glob.glob(os.path.join(src_images_dir, "*.jpeg")) + \
                    glob.glob(os.path.join(src_images_dir, "*.png"))

        for img_path in img_files:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            src_label_path = os.path.join(src_labels_dir, base_name + ".txt")
            dest_label_path = os.path.join(dest_labels_dir, base_name + ".txt")
            dest_img_path = os.path.join(dest_images_dir, img_name)
            
            if not os.path.exists(src_label_path):
                continue
                
            valid_lines = []
            with open(src_label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                    w, h = float(parts[3]), float(parts[4])
                    area = w * h
                    
                    # Filtering criteria: Width <= 95%, Height <= 95%, Area <= 90%
                    if w < 0.95 and h < 0.95 and area < 0.90:
                        valid_lines.append(line)
                        stats["annotations_kept"] += 1
                    else:
                        stats["annotations_discarded"] += 1

            if valid_lines:
                # Keep image and valid labels
                shutil.copy(img_path, dest_img_path)
                with open(dest_label_path, "w") as f:
                    f.writelines(valid_lines)
                stats["images_kept"] += 1
            else:
                # Discard image completely if no valid labels remain
                stats["images_discarded"] += 1

    # Copy data.yaml
    with open(os.path.join(dest_dir, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f)

    print("Dataset Cleaning Complete.")
    print(f"Images Kept: {stats['images_kept']}")
    print(f"Images Discarded: {stats['images_discarded']}")
    print(f"Annotations Kept: {stats['annotations_kept']}")
    print(f"Annotations Discarded: {stats['annotations_discarded']}")

if __name__ == '__main__':
    clean_dataset()
