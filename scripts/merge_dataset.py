import os
import shutil
from pathlib import Path

def merge_datasets(source_dir, target_dir):
    print(f"Starting merge from {source_dir} to {target_dir}")
    
    # Mapping logic based on data.yaml
    mapping = {
        0: 1, 1: 1, 2: 1,  # broken_rail -> crack (1)
        3: 2, 4: 2, 5: 2,  # corrosion -> flaking (2)
        6: 4, 7: 4, 8: 4,  # head_checks -> sheling (4)
        9: 3, 10: 3, 11: 3, # join_bar -> joints (3)
        12: 5, 13: 5, 14: 5 # squats -> spalling (5)
    }

    splits = ["train", "valid", "test"]
    
    total_images_copied = 0
    total_labels_copied = 0

    for split in splits:
        src_img_dir = os.path.join(source_dir, split, "images")
        src_lbl_dir = os.path.join(source_dir, split, "labels")
        
        if not os.path.exists(src_img_dir) or not os.path.exists(src_lbl_dir):
            print(f"Skipping {split} - folders not found.")
            continue

        tgt_img_dir = os.path.join(target_dir, split, "images")
        tgt_lbl_dir = os.path.join(target_dir, split, "labels")
        
        os.makedirs(tgt_img_dir, exist_ok=True)
        os.makedirs(tgt_lbl_dir, exist_ok=True)

        images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            lbl_name = base_name + ".txt"
            
            src_img_path = os.path.join(src_img_dir, img_name)
            src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
            
            # If the image has no label file, we skip it (clean data!)
            if not os.path.exists(src_lbl_path):
                continue
            
            # Prefix to avoid naming collisions
            new_img_name = f"v2_roboflow_{img_name}"
            new_lbl_name = f"v2_roboflow_{lbl_name}"
            
            tgt_img_path = os.path.join(tgt_img_dir, new_img_name)
            tgt_lbl_path = os.path.join(tgt_lbl_dir, new_lbl_name)
            
            # Read label, map classes, write new label
            valid_boxes = False
            with open(src_lbl_path, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in mapping:
                        new_cls_id = mapping[cls_id]
                        new_lines.append(f"{new_cls_id} {' '.join(parts[1:])}\n")
                        valid_boxes = True
            
            # Only copy image if it has valid boxes after mapping
            if valid_boxes:
                # write new label
                with open(tgt_lbl_path, "w") as f:
                    f.writelines(new_lines)
                
                # copy image
                shutil.copy2(src_img_path, tgt_img_path)
                
                total_images_copied += 1
                total_labels_copied += 1

    print(f"\nMerge Complete!")
    print(f"Successfully migrated {total_images_copied} images and {total_labels_copied} labels into dataset_clean.")

if __name__ == "__main__":
    SOURCE = r"C:\Users\ABHISHEK\ML project\railway-defect-mlops\raw_v2_dataset"
    TARGET = r"C:\Users\ABHISHEK\ML project\railway-defect-mlops\dataset_clean"
    merge_datasets(SOURCE, TARGET)
