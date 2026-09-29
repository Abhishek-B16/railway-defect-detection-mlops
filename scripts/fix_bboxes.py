import os
import glob

def clip_yolo_bbox(x, y, w, h):
    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2
    
    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))
    
    new_w = x_max - x_min
    new_h = y_max - y_min
    new_x = x_min + new_w / 2
    new_y = y_min + new_h / 2
    
    return new_x, new_y, new_w, new_h

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
                        nx, ny, nw, nh = clip_yolo_bbox(x, y, w, h)
                        if nw > 0 and nh > 0:
                            new_lines.append(f"{c} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
                            modified = True
                        else:
                            # Box collapsed completely, drop it
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
