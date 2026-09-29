import yaml
import os
import glob
from collections import defaultdict
import json

def audit_dataset(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    names = data.get('names', [])
    nc = data.get('nc', 0)
    
    print("=== Phase 1: Dataset Audit ===")
    print(f"Classes ({nc}): {names}")
    
    base_dir = os.path.dirname(yaml_path)
    splits = ['train', 'valid', 'test']
    
    class_counts = defaultdict(int)
    total_images = 0
    total_labels = 0
    
    issues = {
        'empty_labels': 0,
        'missing_images': 0,
        'missing_labels': 0,
        'invalid_class_ids': 0,
        'out_of_bounds_boxes': 0,
        'tiny_boxes': 0, # width or height < 0.01 (1%)
        'huge_boxes': 0  # width or height > 0.95 (95%)
    }
    
    for split in splits:
        img_dir = os.path.join(base_dir, split, 'images')
        lbl_dir = os.path.join(base_dir, split, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue
            
        imgs = set(os.path.basename(p) for p in glob.glob(os.path.join(img_dir, '*.*')))
        lbls = set(os.path.basename(p) for p in glob.glob(os.path.join(lbl_dir, '*.txt')))
        
        split_imgs = len(imgs)
        total_images += split_imgs
        
        print(f"Split '{split}': {split_imgs} images")
        
        for img in imgs:
            name, _ = os.path.splitext(img)
            lbl_name = name + '.txt'
            lbl_path = os.path.join(lbl_dir, lbl_name)
            
            if not os.path.exists(lbl_path):
                issues['missing_labels'] += 1
                continue
                
            with open(lbl_path, 'r') as lf:
                lines = lf.readlines()
                
            if not lines:
                issues['empty_labels'] += 1
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                total_labels += 1
                try:
                    c = int(parts[0])
                    if c < 0 or c >= nc:
                        issues['invalid_class_ids'] += 1
                    else:
                        class_counts[names[c]] += 1
                        
                    x, y, w, h = map(float, parts[1:])
                    if x < 0 or y < 0 or x > 1 or y > 1 or (x+w/2) > 1 or (y+h/2) > 1 or (x-w/2) < 0 or (y-h/2) < 0:
                        issues['out_of_bounds_boxes'] += 1
                    
                    if w < 0.01 or h < 0.01:
                        issues['tiny_boxes'] += 1
                    if w > 0.95 or h > 0.95:
                        issues['huge_boxes'] += 1
                        
                except ValueError:
                    pass
                    
    print(f"\nTotal Images: {total_images}")
    print(f"Total Annotations: {total_labels}")
    
    print("\nClass Distribution:")
    for c in names:
        count = class_counts[c]
        pct = (count / total_labels * 100) if total_labels > 0 else 0
        print(f"  - {c}: {count} annotations ({pct:.2f}%)")
        
    print("\nQuality Issues:")
    for k, v in issues.items():
        print(f"  - {k}: {v}")

if __name__ == '__main__':
    audit_dataset('C:/Users/ABHISHEK/ML project/railway-defect-mlops/dataset_clean/data.yaml')
