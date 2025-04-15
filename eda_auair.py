# eda_auair.py  ── quick EDA for AU‑AIR style annotations
# -------------------------------------------------------
import json
from pathlib import Path
import random

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------------------
# 1) Paths
ANN_PATH = Path("AU-AIR/annotations.json")
IMG_DIR  = Path("AU-AIR/images")
PLOT_DIR = "eda_plots"

# -------------------------------------------------------------------
# 2) Load annotation file
raw = json.loads(ANN_PATH.read_text())

records = raw["annotations"]          # list of frames
cat_names = raw["categories"]         # list of class names

# -------------------------------------------------------------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/{name}.png", dpi=150)
    plt.close(fig)
# 3) Flatten into a DataFrame of one row per bounding‑box
rows = []
for frame in records:
    # fix the colon typo once:
    w = frame.get("image_width", frame.get("image_width:", None))
    h = frame["image_height"]
    img_name = frame["image_name"]
    for bb in frame["bbox"]:
        rows.append({
            "image":   img_name,
            "img_w":   int(w),
            "img_h":   int(h),
            "x":       bb["left"],
            "y":       bb["top"],
            "w":       bb["width"],
            "h":       bb["height"],
            "cls_id":  bb["class"],
            "cls":     cat_names[bb["class"]],
        })

df = pd.DataFrame(rows)

# -------------------------------------------------------------------
# 4) Basic dataset facts
n_imgs = df["image"].nunique()
n_boxes = len(df)
print(f"Images: {n_imgs:,}\nBoxes : {n_boxes:,}\nClasses: {cat_names}")

# -------------------------------------------------------------------
# 5) Class distribution
cls_counts = df["cls"].value_counts().sort_values(ascending=False)
print("\n=== Boxes per class ===")
print(cls_counts)

plt.figure()
cls_counts.plot(kind="bar")
plt.title("Class distribution (bounding‑boxes)")
plt.xlabel("class"); plt.ylabel("count")
plt.tight_layout(); plt.show()

# -------------------------------------------------------------------
# 6) Boxes per image
boxes_per_img = df.groupby("image").size()
print("\nBoxes per image  (min / mean / max):",
      boxes_per_img.min(), boxes_per_img.mean().round(2), boxes_per_img.max())

plt.figure()
boxes_per_img.hist(bins=range(1, boxes_per_img.max()+2))
plt.title("Boxes per image")
plt.xlabel("#boxes"); plt.ylabel("#images")
plt.tight_layout(); plt.show()

# -------------------------------------------------------------------
# 7) Geometry statistics
df["area"]  = df["w"] * df["h"]
df["ar"]    = df["w"] / df["h"]

for col, bins, xlab in [
        ("w",   40, "box width (px)"),
        ("h",   40, "box height (px)"),
        ("area",40, "box area (px²)"),
        ("ar",  40, "aspect ratio  w/h")]:
    plt.figure()
    df[col].hist(bins=bins)
    plt.title(f"Histogram of {col}")
    plt.xlabel(xlab); plt.ylabel("count")
    plt.tight_layout(); plt.show()

# -------------------------------------------------------------------
# 8) Image resolution(s)
resolutions = df[["img_w","img_h"]].drop_duplicates()
print("\nUnique image resolutions:")
print(resolutions)

# -------------------------------------------------------------------
# 9) Visual sanity check – show 9 random frames with boxes
sample_imgs = random.sample(records, k=9)

ncols = 3
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()

for ax, frame in zip(axes, sample_imgs):
    img_path = IMG_DIR / frame["image_name"]
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img)
    for bb in frame["bbox"]:
        rect = patches.Rectangle(
            (bb["left"], bb["top"]),
            bb["width"], bb["height"],
            linewidth=1, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(bb["left"], bb["top"]-2,
                cat_names[bb["class"]], color="lime", fontsize=6)
    ax.set_axis_off()
plt.suptitle("Random AU‑AIR samples with bounding‑boxes", y=0.92)
plt.tight_layout(); plt.show()

