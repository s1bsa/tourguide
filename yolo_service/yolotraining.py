import os
from datasets import load_dataset
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image
from collections import Counter

#load dataset that contains artifacts
dataset = load_dataset("firdevsnamal/museum_dataset")

#extract department names for class labels 
dept_names = dataset["train"].features["department"].names

#combine department + object_name to allow for higher detail classification to feed into the LLM 
def make_label(sample):
    dept = dept_names[sample["department"]]
    obj = sample["object_name"].strip().title() if sample["object_name"] else "Unknown"
    combined = f"{dept}_{obj}".replace("/", "_").replace("\\", "_").replace(" ", "_")
    return combined


all_labels = [make_label(s) for s in dataset["train"]]

#filter out rare instances to make training more efficient 
min_samples = 10
counts = Counter(all_labels)
filtered_labels = [lbl for lbl, c in counts.items() if c >= min_samples]

print(f"✅ Found {len(filtered_labels)} valid combined classes after filtering.")

#folder structure 
base_dir = "museum_combined_dataset"
splits = ["train", "val", "test"]

for split in splits:
    for label in filtered_labels:
        os.makedirs(os.path.join(base_dir, split, label), exist_ok=True)


def save_split(split_name, split_data):
    for i, sample in enumerate(tqdm(split_data, desc=f"Preparing {split_name} split")):
        label = make_label(sample)
        if label not in filtered_labels:
            continue

        img = sample["image"]
        label_dir = os.path.join(base_dir, split_name, label)
        os.makedirs(label_dir, exist_ok=True)

        img_path = os.path.join(label_dir, f"{i}.jpg")
        img.save(img_path)

if "train" in dataset:
    save_split("train", dataset["train"])
if "validation" in dataset:
    save_split("val", dataset["validation"])
if "test" in dataset:
    save_split("test", dataset["test"])


#using classification version of yolo as we do not care for bounding boxes
model = YOLO("yolo11n-cls.pt")

#train yolo, using CPU systems due to system requirements 
model.train(
    data=base_dir,
    epochs=30,           
    imgsz=224,           
    batch=8,             
    workers=0,           
    device="cpu",        
    name="museum_combined_cpu_train"
)
