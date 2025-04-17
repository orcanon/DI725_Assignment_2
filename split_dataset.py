import json
import random
from pathlib import Path
from shutil import copy2

# --- User configuration ---
INPUT_IMAGES_DIR = Path("AU-AIR/images")  # folder where all images are now
INPUT_ANNOTATION_FILE = Path("auair_coco.json")
OUTPUT_DIR = Path("coco_format")
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
RANDOM_SEED = 42

# --- Load COCO data ---
with open(INPUT_ANNOTATION_FILE, 'r') as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]
licenses = coco.get("licenses", [])
info = coco.get("info", {})

# --- Shuffle and split ---
random.seed(RANDOM_SEED)
random.shuffle(images)

n = len(images)
n_train = int(SPLIT_RATIOS["train"] * n)
n_val = int(SPLIT_RATIOS["val"] * n)
n_test = n - n_train - n_val

splits = {
    "train": images[:n_train],
    "val": images[n_train:n_train+n_val],
    "test": images[n_train+n_val:]
}

# --- Create output dirs ---
(OUTPUT_DIR / "annotations").mkdir(parents=True, exist_ok=True)
for split in splits:
    (OUTPUT_DIR / f"{split}2017").mkdir(parents=True, exist_ok=True)

# --- Process each split ---
for split, split_images in splits.items():
    image_ids = {img["id"] for img in split_images}
    split_annos = [a for a in annotations if a["image_id"] in image_ids]

    split_dict = {
        "info": info,
        "licenses": licenses,
        "images": split_images,
        "annotations": split_annos,
        "categories": categories
    }

    # Save new annotation file
    out_json = OUTPUT_DIR / "annotations" / f"custom_{split}.json"
    with open(out_json, 'w') as f:
        json.dump(split_dict, f, indent=2)

    # Copy images
    for img in split_images:
        src_path = INPUT_IMAGES_DIR / img["file_name"]
        dst_path = OUTPUT_DIR / f"{split}2017" / img["file_name"]
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.exists():
            copy2(src_path, dst_path)
        else:
            print(f"[WARNING] Missing image: {src_path}")
