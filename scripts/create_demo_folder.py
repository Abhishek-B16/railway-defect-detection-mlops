import os
import glob
import shutil
import requests

API_URL = "http://localhost:8000/predict"
OUTPUT_DIR = "demo_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ["crack", "bolts", "joints", "flaking", "sheling", "spalling"]
CLASS_INDEX = {
    "bolts": 0,
    "crack": 1,
    "flaking": 2,
    "joints": 3,
    "sheling": 4,
    "spalling": 5
}

def predict_image(path):
    try:
        with open(path, "rb") as f:
            r = requests.post(API_URL, files={"file": (os.path.basename(path), f, "image/jpeg")}, timeout=10)
        if r.status_code == 200:
            return r.json().get("detections", [])
    except Exception as e:
        print(f"Error predicting {path}: {e}")
    return None

print("=== 1. Finding 'No Defect' (Clean Track) Images ===")
# Search in new_cracks_dataset for empty labels
empty_txts = [t for t in glob.glob("new_cracks_dataset/**/*.txt", recursive=True) if os.path.getsize(t) == 0]
no_defect_selected = []
for t in empty_txts:
    img = t.replace(".txt", ".jpg")
    if not os.path.exists(img):
        img = t.replace(".txt", ".png")
    if os.path.exists(img):
        dets = predict_image(img)
        if dets is not None and len(dets) == 0:
            no_defect_selected.append(img)
            print(f"Found clean track image: {os.path.basename(img)}")
            if len(no_defect_selected) >= 3:
                break

for idx, path in enumerate(no_defect_selected, 1):
    dest = os.path.join(OUTPUT_DIR, f"00_no_defect_sample_{idx}.jpg")
    shutil.copyfile(path, dest)
    print(f"Copied -> {dest}")

print("\n=== 2. Finding Confirmed Defect Images for each class ===")
for c in CLASSES:
    target_idx = CLASS_INDEX[c]
    selected = []
    print(f"\nLooking for class: {c} (ID {target_idx})...")
    
    # Check test set first, then valid set
    label_files = glob.glob("dataset_clean/test/labels/*.txt") + glob.glob("dataset_clean/valid/labels/*.txt")
    for lf in label_files:
        with open(lf, "r") as f:
            lines = [line.strip().split() for line in f if line.strip()]
        if any(int(parts[0]) == target_idx for parts in lines):
            base_name = os.path.basename(lf).replace(".txt", ".jpg")
            img_dir = os.path.dirname(lf).replace("labels", "images")
            img_path = os.path.join(img_dir, base_name)
            if not os.path.exists(img_path):
                img_path = img_path.replace(".jpg", ".png")
            if os.path.exists(img_path):
                dets = predict_image(img_path)
                if dets:
                    matching = [d for d in dets if d["class_name"] == c]
                    if matching:
                        conf = matching[0]["confidence"]
                        selected.append((img_path, conf))
                        print(f"  [+] Match found: {base_name} (Confidence: {conf*100:.1f}%)")
                        if len(selected) >= 3:
                            break

    # If test/valid had fewer than 3, just use the ground truth images
    if len(selected) < 3:
        for lf in label_files:
            with open(lf, "r") as f:
                lines = [line.strip().split() for line in f if line.strip()]
            if any(int(parts[0]) == target_idx for parts in lines):
                base_name = os.path.basename(lf).replace(".txt", ".jpg")
                img_dir = os.path.dirname(lf).replace("labels", "images")
                img_path = os.path.join(img_dir, base_name)
                if os.path.exists(img_path) and img_path not in [s[0] for s in selected]:
                    selected.append((img_path, 0.5))
                    if len(selected) >= 3:
                        break

    for idx, (path, conf) in enumerate(selected, 1):
        dest = os.path.join(OUTPUT_DIR, f"{c}_sample_{idx}.jpg")
        shutil.copyfile(path, dest)
        print(f"Saved: {dest} (Conf: {conf*100:.1f}%)")

print("\n=== Demo samples created successfully in 'demo_samples/' ===")
