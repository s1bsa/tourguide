import os
from ultralytics import YOLO

base_dir = os.path.dirname(__file__)  # the directory where this script lives
model_path = os.path.join(base_dir, "museum_combined_cpu_train", "weights", "best.pt")

model = YOLO(model_path)

result = model.predict("artifact.jpg")[0]
pred_label = result.names[result.probs.top1]

# Split back into department and object
dept, obj = pred_label.split("_", 1)
print(f"Department: {dept.replace('_', ' ')}")
print(f"Object: {obj.replace('_', ' ')}")
