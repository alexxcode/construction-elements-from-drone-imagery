"""Generate the README figures from the dataset and the saved training results.

Run from the repository root:
    python scripts/make_figures.py
"""
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
RUN = ROOT / "models" / "zanjas_cableado_20250902_035309"
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)

NAMES = yaml.safe_load(open(DATA / "data.yaml", encoding="utf-8"))["names"]
# English names for the original (Spanish) class labels
NAMES_EN = [
    "Pole", "Access gate", "Bird-protection plate", "Fence mesh", "Wildlife crossing",
    "Trench protection plates", "Junction boxes", "Silt", "MV cable", "Container",
    "Trench", "Trench guide spray", "Cable reels", "Corrugated conduit",
    "Bare copper cable", "Yellow electrical-hazard tape", "LV cable",
]
SPLITS = ("train", "val", "test")
PALETTE = list(plt.get_cmap("tab20").colors)
# clearly distinguishable colours for the classes that have annotations
for cls, hexc in {5: "#2ecc40", 7: "#ff851b", 10: "#0074d9", 11: "#f012be",
                  13: "#ffdc00", 15: "#ff4136", 16: "#39cccc"}.items():
    PALETTE[cls] = tuple(int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))


def color(cls):
    r, g, b = PALETTE[cls % len(PALETTE)]
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def read_labels(txt):
    items = []
    for line in open(txt, encoding="utf-8"):
        v = line.split()
        if len(v) >= 7:
            items.append((int(v[0]), np.array(v[1:], float).reshape(-1, 2)))
    return items


def draw_masks(img, labels, alpha=0.45):
    h, w = img.shape[:2]
    overlay = img.copy()
    for cls, pts in labels:
        poly = (pts * [w, h]).astype(np.int32)
        cv2.fillPoly(overlay, [poly], color(cls))
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    for cls, pts in labels:
        poly = (pts * [w, h]).astype(np.int32)
        cv2.polylines(out, [poly], True, color(cls), max(2, w // 600))
    return out


def legend_handles(classes):
    from matplotlib.patches import Patch
    return [Patch(color=PALETTE[c % len(PALETTE)], label=NAMES_EN[c]) for c in sorted(classes)]


# 1. Training curves ----------------------------------------------------------
def training_curves():
    df = pd.read_csv(RUN / "results.csv")
    df.columns = df.columns.str.strip()
    # Ultralytics segmentation fitness (the criterion used to save best.pt)
    fit = sum(0.1 * df[f"metrics/mAP50({t})"] + 0.9 * df[f"metrics/mAP50-95({t})"] for t in "BM")
    best = df.loc[fit.idxmax(), "epoch"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, lab in [("train/seg_loss", "train seg"), ("val/seg_loss", "val seg"),
                   ("train/box_loss", "train box"), ("val/box_loss", "val box")]:
        ax[0].plot(df["epoch"], df[k], label=lab)
    ax[0].set_title("Segmentation and box losses")
    ax[1].plot(df["epoch"], df["val/cls_loss"], label="val cls", color="tab:red")
    ax[1].set_title("Classification loss (validation)")
    for k, lab in [("metrics/mAP50(M)", "mask mAP50"), ("metrics/mAP50-95(M)", "mask mAP50-95"),
                   ("metrics/mAP50(B)", "box mAP50"), ("metrics/mAP50-95(B)", "box mAP50-95")]:
        ax[2].plot(df["epoch"], df[k], label=lab)
    ax[2].axvline(best, ls="--", c="gray", lw=1)
    ax[2].text(best + 0.5, ax[2].get_ylim()[1] * 0.92, f"best.pt\nepoch {best}", fontsize=8)
    ax[2].set_title("Validation metrics")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=130)
    plt.close(fig)


# 2. Instances per class and split --------------------------------------------
def class_distribution():
    counts = {s: Counter() for s in SPLITS}
    for s in SPLITS:
        for txt in (DATA / "labels" / s).glob("*.txt"):
            for cls, _ in read_labels(txt):
                counts[s][cls] += 1
    used = sorted(set().union(*[c.keys() for c in counts.values()]),
                  key=lambda c: -sum(counts[s][c] for s in SPLITS))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    left = np.zeros(len(used))
    for s, col in zip(SPLITS, ("#4C72B0", "#DD8452", "#55A868")):
        vals = np.array([counts[s][c] for c in used])
        ax.barh([NAMES_EN[c] for c in used], vals, left=left, color=col, label=s)
        left += vals
    for i, v in enumerate(left):
        ax.text(v + 0.5, i, int(v), va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("annotated instances")
    ax.set_title("Instances per class (classes with at least one annotation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "class_distribution.png", dpi=130)
    plt.close(fig)


# 3. Mask size per class ------------------------------------------------------
def mask_area():
    areas = defaultdict(list)
    for txt in (DATA / "labels").glob("*/*.txt"):
        for cls, p in read_labels(txt):
            x, y = p[:, 0], p[:, 1]
            areas[cls].append(50 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))  # % of image
    order = sorted(areas, key=lambda c: np.median(areas[c]))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bp = ax.boxplot([areas[c] for c in order], vert=False, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(PALETTE[c]); patch.set_alpha(0.8)
    ax.set_yticks(range(1, len(order) + 1), [f"{NAMES_EN[c]} (n={len(areas[c])})" for c in order])
    ax.set_xscale("log")
    ax.set_xlabel("mask area (% of the 4000×3000 image, log scale)")
    ax.set_title("Object size per class")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "mask_area.png", dpi=130)
    plt.close(fig)


# 4. Ground-truth mosaic ------------------------------------------------------
def annotation_mosaic(n=6, width=1000):
    txts = [t for t in (DATA / "labels" / "train").glob("*.txt") if read_labels(t)]
    # first the images that add new classes, then the most annotated ones
    chosen, seen = [], set()
    for t in sorted(txts, key=lambda p: -len({c for c, _ in read_labels(p)})):
        cl = {c for c, _ in read_labels(t)}
        if cl - seen:
            chosen.append(t); seen |= cl
    for t in sorted(txts, key=lambda p: -len(read_labels(p))):
        if len(chosen) >= n:
            break
        if t not in chosen:
            chosen.append(t); seen |= {c for c, _ in read_labels(t)}
    chosen = sorted(chosen[:n])
    tiles = []
    for t in chosen:
        img = cv2.imread(str(DATA / "images" / "train" / (t.stem + ".jpg")))
        img = draw_masks(img, read_labels(t))
        tiles.append(cv2.resize(img, (width, int(width * img.shape[0] / img.shape[1]))))
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    for a, tile, t in zip(axes.flat, tiles, chosen):
        a.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)); a.set_title(t.stem, fontsize=9); a.axis("off")
    fig.legend(handles=legend_handles(seen), loc="lower center", ncol=4, fontsize=11, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "annotations_mosaic.jpg", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    training_curves()
    class_distribution()
    mask_area()
    annotation_mosaic()
    print("Figures saved to", OUT)
