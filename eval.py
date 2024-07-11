from ultralytics import YOLO

# Load the model.
model = YOLO('/mnt/d/task/models/')

# Evaluate the model.
results = model.val(
    data='./shelves_data/data.yaml',
    imgsz=640,
    name='yolov8m_eval', 
)

# Extract and print evaluation metrics.
metrics = results.metrics
precision = metrics['precision']
recall = metrics['recall']
f1_score = metrics['f1']
map50 = metrics['map50']  # Mean Average Precision at IoU=0.50
map = metrics['map']      # Mean Average Precision at IoU=0.50:0.95

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")
print(f"mAP@0.50: {map50:.4f}")
print(f"mAP@0.50:0.95: {map:.4f}")