import os
import glob

def fix_dataset_bboxes(dataset_dir):
    splits = ['train', 'valid', 'test']
    fixed_count = 0
    
    for split in splits:
        lbl_dir = os.path.join(dataset_dir, split, 'labels')
        if not os.path.exists(lbl_dir):
            continue
            
        for txt_file in glob.glob(os.path.join(lbl_dir, '*.txt')):
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            modified = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    c = parts[0]
                    x, y, w, h = map(float, parts[1:])
                    
                    if x < 0 or y < 0 or x > 1 or y > 1 or (x+w/2) > 1 or (y+h/2) > 1 or (x-w/2) < 0 or (y-h/2) < 0:
                        x_min = max(0.000001, x - w / 2.0)
                        y_min = max(0.000001, y - h / 2.0)
                        x_max = min(0.999999, x + w / 2.0)
                        y_max = min(0.999999, y + h / 2.0)
                        
                        nx = (x_min + x_max) / 2.0
                        ny = (y_min + y_max) / 2.0
                        nw = x_max - x_min
                        nh = y_max - y_min
                        
                        if nw > 0 and nh > 0:
                            new_lines.append(f"{c} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
                            modified = True
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
                    
            if modified:
                with open(txt_file, 'w') as f:
                    f.writelines(new_lines)
                fixed_count += 1
                
    print(f"Fixed out-of-bounds boxes in {fixed_count} label files.")

if __name__ == '__main__':
    fix_dataset_bboxes('C:/Users/ABHISHEK/ML project/railway-defect-mlops/dataset_clean')
