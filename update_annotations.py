import os
import glob

def update_annotations(path):
    annotation_files = glob.glob(os.path.join(path, '*.txt'))
    
    for file in annotation_files:
        with open(file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            parts[0] = '0'  # Change the class ID to 0
            new_lines.append(' '.join(parts))
        
        with open(file, 'w') as f:
            f.write('\n'.join(new_lines))

# Paths to your annotation directories
train_annotation_path = '/mnt/d/task/shelves_data/train/labels'
val_annotation_path = '/mnt/d/task/shelves_data/valid/labels'
test_annotation_path = '/mnt/d/task/shelves_data/test/labels'

# Update annotations for train, val, and test sets
update_annotations(train_annotation_path)
update_annotations(val_annotation_path)
update_annotations(test_annotation_path)

print("Annotations updated successfully.")