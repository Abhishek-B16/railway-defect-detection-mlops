import os
import glob

def clean_old_cracks():
    target_dir = r"C:\Users\ABHISHEK\ML project\railway-defect-mlops\dataset_clean"
    
    deleted_old_cracks = 0
    kept_hd_cracks = 0
    
    for split in ["train", "valid", "test"]:
        lbl_dir = os.path.join(target_dir, split, "labels")
        if not os.path.exists(lbl_dir): continue
        
        txt_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        
        for txt_path in txt_files:
            filename = os.path.basename(txt_path)
            
            with open(txt_path, "r") as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                class_id = int(parts[0])
                
                # If it's class 1 (crack)
                if class_id == 1:
                    if filename.startswith("v5_HD_"):
                        # Keep it! It's our clean new dataset
                        new_lines.append(line)
                        kept_hd_cracks += 1
                    else:
                        # DELETE IT! It's the old messy dataset
                        deleted_old_cracks += 1
                else:
                    # Keep all other classes (bolts, joints, etc)
                    new_lines.append(line)
                    
            # Overwrite the text file
            with open(txt_path, "w") as f:
                f.writelines(new_lines)

    print(f"✅ Surgically removed {deleted_old_cracks} old messy crack boxes!")
    print(f"✅ Kept {kept_hd_cracks} perfect HD crack boxes!")
    print("Your dataset is now perfectly clean and ready for V5.")

if __name__ == "__main__":
    clean_old_cracks()
