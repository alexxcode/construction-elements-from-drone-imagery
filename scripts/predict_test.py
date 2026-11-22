"""Run the trained model on the annotated validation and test images and
compare ground truth against predictions.

Run from the repository root:
    python scripts/predict_test.py [--conf 0.02]
"""
import argparse

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from make_figures import DATA, OUT, ROOT, draw_masks, legend_handles, read_labels

WEIGHTS = ROOT / "models" / "zanjas_cableado_20250902_035309" / "best.pt"


def main(conf):
    model = YOLO(str(WEIGHTS))
    imgs = [p for split in ("val", "test") for p in sorted((DATA / "images" / split).glob("*.jpg"))
            if (DATA / "labels" / split / (p.stem + ".txt")).stat().st_size > 0]
    rows, classes = [], set()
    for p in imgs:
        img = cv2.imread(str(p))
        gt = read_labels(DATA / "labels" / p.parent.name / (p.stem + ".txt"))
        r = model.predict(str(p), imgsz=1024, conf=conf, retina_masks=True, verbose=False)[0]
        pred = []
        if r.masks is not None:
            for cls, poly in zip(r.boxes.cls.int().tolist(), r.masks.xyn):
                if len(poly) >= 3:
                    pred.append((cls, poly))
        classes |= {c for c, _ in gt} | {c for c, _ in pred}
        w = 900
        size = (w, int(w * img.shape[0] / img.shape[1]))
        rows.append((f"{p.parent.name} · {p.stem}", cv2.resize(draw_masks(img, gt), size),
                     cv2.resize(draw_masks(img, pred), size), len(gt), len(pred)))
        print(f"{p.name}: {len(gt)} ground-truth instances, {len(pred)} predictions")

    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4.6 * len(rows)))
    for (title, g, pr, ng, npred), (a, b) in zip(rows, axes):
        a.imshow(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)); a.set_title(f"{title} · ground truth ({ng})", fontsize=9)
        b.imshow(cv2.cvtColor(pr, cv2.COLOR_BGR2RGB)); b.set_title(f"prediction, conf ≥ {conf} ({npred})", fontsize=9)
        a.axis("off"); b.axis("off")
    fig.legend(handles=legend_handles(classes), loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "gt_vs_pred.jpg", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.02)
    main(ap.parse_args().conf)
