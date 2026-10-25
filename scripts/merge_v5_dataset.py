import os
import shutil
import glob

def merge_v5_dataset():
    source_dir = r"C:\Users\ABHISHEK\ML project\railway-defect-mlops\v5_raw_dataset"
    target_dir = r"C:\Users\ABHISHEK\ML project\railway-defect-mlops\dataset_clean"
    
    if not os.path.exists(source_dir):
        print(f"Error: Could not find {source_dir}. Make sure you extracted it there!")
        return

    # In our MASTER dataset (dataset_clean):
    # 0=bolts, 1=crack, 2=flaking, 3=joints, 4=sheling, 5=spalling
    
    # In the NEW dataset (v5_raw_dataset):
    # 0=No Fault, 1=fault (crack)
    
    # STRATEGY:
    # 1. We ignore/delete any bounding boxes that say '0' (No Fault) because YOLO will think it's a 'bolt'!
    # 2. We keep any bounding boxes that say '1' (fault) because '1' matches 'crack' in our master dataset perfectly!
    
    for split in ["train", "valid", "test"]:
        source_img_dir = os.path.join(source_dir, split, "images")
        source_lbl_dir = os.path.join(source_dir, split, "labels")
        
        target_img_dir = os.path.join(target_dir, split, "images")
        target_lbl_dir = os.path.join(target_dir, split, "labels")
        
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)
        
        if not os.path.exists(source_lbl_dir):
            continue
            
        txt_files = glob.glob(os.path.join(source_lbl_dir, "*.txt"))
        
        print(f"Processing {len(txt_files)} files in {split}...")
        
        for txt_path in txt_files:
            filename = os.path.basename(txt_path)
            img_filename = filename.replace(".txt", ".jpg") 
            
            img_path = os.path.join(source_img_dir, img_filename)
            
            # Read the labels
            valid_lines = []
            with open(txt_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                    class_id = int(parts[0])
                    
                    if class_id == 1:
                        # Keep it! It's a crack (class 1)
                        valid_lines.append(line)
                    # If it's class_id == 0, we completely ignore it.
            
            # Only copy the image and text file if there's actually a crack in it!
            if len(valid_lines) > 0 and os.path.exists(img_path):
                # Save new text file to master dataset
                target_txt_path = os.path.join(target_lbl_dir, "v5_HD_" + filename)
                with open(target_txt_path, "w") as f:
                    f.writelines(valid_lines)
                    
                # Copy image to master dataset
                target_img_path = os.path.join(target_img_dir, "v5_HD_" + img_filename)
                shutil.copy(img_path, target_img_path)

    print("✅ SUCCESS! HD Images injected perfectly into the master dataset without confusing the AI!")

if __name__ == "__main__":
    merge_v5_dataset()
