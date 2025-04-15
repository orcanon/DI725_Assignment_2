"""
Convert AU‑AIR annotations.json  ➜  COCO style JSON for DETR.
"""
import json, itertools
from pathlib import Path

src = Path("AU-AIR/annotations.json").read_text()
data = json.loads(src)

# --- 1. build the COCO 'images' list --------------------------
images, annots = [], []
ann_id = itertools.count(1)     # running object id

for img_id, frame in enumerate(data["annotations"], start=1):
    # fix the colon typo once:
    w = int(frame.get("image_width", frame.get("image_width:", 0)))
    h = int(frame["image_height"])

    images.append({
        "id": img_id,
        "file_name": frame["image_name"],
        "width":  w,
        "height": h
    })

    for box in frame["bbox"]:
        x, y, w_box, h_box = box["left"], box["top"], box["width"], box["height"]
        annots.append({
            "id": next(ann_id),
            "image_id": img_id,
            "category_id": box["class"],
            "bbox": [x, y, w_box, h_box],
            "area": w_box * h_box,
            "iscrowd": 0
        })

# --- 2. turn category names into COCO objects -----------------
categories = [
    {"id": i, "name": name, "supercategory": "object"}
    for i, name in enumerate(data["categories"])
]

coco = {"info": data["info"],
        "licenses": data["licenses"],
        "images": images,
        "annotations": annots,
        "categories": categories}

Path("auair_coco.json").write_text(json.dumps(coco, indent=2))
print("✅  Wrote auair_coco.json")
