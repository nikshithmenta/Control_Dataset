"""Generate a shape-vs-texture control dataset.

Each source object image is turned into three aligned 224x224 conditions:

  original      both cues present (baseline)
  edges         Canny edge map: shape kept, texture removed
  texture_patch kxk grid shuffled: texture kept, global shape destroyed

The only thing that changes between conditions is which cue survives, so any
difference in a classifier's accuracy across conditions is attributable to that
classifier's reliance on shape vs texture.

Motivated by Geirhos et al., "ImageNet-trained CNNs are biased towards texture;
increasing shape bias improves accuracy and robustness", ICLR 2019
(arXiv:1811.12231). The "edges" condition follows their Appendix A.6 recipe; the
"texture_patch" condition is the scrambled-shape / texture manipulation they cite
(Gatys et al. 2017; Brendel & Bethge 2019).

Source images come from torchvision's Imagenette (10 ImageNet classes), so a
standard ImageNet-trained ResNet-50 can be evaluated on them directly. Pass
--input-dir to use your own image folder instead of downloading.
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SIZE = 224
CONDITIONS = ("original", "edges", "texture_patch")

# Imagenette class folder -> human-readable label. The folder names are WordNet
# synset ids; these are the 10 Imagenette classes.
IMAGENETTE_LABELS = {
    "n01440764": "tench",
    "n02102040": "english_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "french_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}


def to_square_rgb(img: Image.Image, size: int = SIZE) -> np.ndarray:
    """Center-crop to square, resize to size, return HxWx3 uint8 RGB."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.BILINEAR)
    return np.asarray(img)


def make_edges(rgb: np.ndarray) -> np.ndarray:
    """Canny edges on a blurred greyscale image, black lines on white.

    Mirrors the paper's MATLAB recipe:
        1 - edge(imgaussfilt(rgb2gray(img), 2), 'Canny')
    """
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grey, ksize=(0, 0), sigmaX=2.0)
    edges = cv2.Canny(blurred, 50, 150)          # white edges on black
    inverted = 255 - edges                        # black edges on white
    return np.repeat(inverted[:, :, None], 3, axis=2)


def make_texture_patch(rgb: np.ndarray, k: int, rng: random.Random) -> np.ndarray:
    """Cut into a kxk grid and randomly permute the patches.

    Local texture statistics survive; the global object contour does not.
    """
    h, w, _ = rgb.shape
    ph, pw = h // k, w // k
    blocks = [
        rgb[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw].copy()
        for i in range(k) for j in range(k)
    ]
    order = list(range(len(blocks)))
    # reshuffle until at least one block moves (avoids the identity permutation)
    while True:
        rng.shuffle(order)
        if any(order[i] != i for i in range(len(order))):
            break
    blocks = [blocks[o] for o in order]
    rows = [np.concatenate(blocks[r * k:(r + 1) * k], axis=1) for r in range(k)]
    return np.concatenate(rows, axis=0)


def iter_sources(args):
    """Yield (label, index, rgb_uint8) for the source images."""
    if args.input_dir:
        root = Path(args.input_dir)
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            label = class_dir.name
            files = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )[: args.per_class]
            for idx, fp in enumerate(files):
                yield label, idx, to_square_rgb(Image.open(fp))
    else:
        from torchvision.datasets import Imagenette
        ds = Imagenette(root=args.cache, split="val", size="320px", download=_need_download(args.cache))
        per_label = {}
        for path, _ in ds._samples:                       # (filepath, class_index)
            synset = Path(path).parent.name
            label = IMAGENETTE_LABELS.get(synset, synset)
            if per_label.get(label, 0) >= args.per_class:
                continue
            idx = per_label.get(label, 0)
            per_label[label] = idx + 1
            yield label, idx, to_square_rgb(Image.open(path))


def _need_download(cache: str) -> bool:
    return not (Path(cache) / "imagenette2-320").exists()


def save(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def build_montage(samples, out_path: Path):
    """samples: list of (label, {condition: rgb}). One row per sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(samples)
    fig, axes = plt.subplots(n, len(CONDITIONS), figsize=(3 * len(CONDITIONS), 3 * n))
    if n == 1:
        axes = axes[None, :]
    for r, (label, imgs) in enumerate(samples):
        for c, cond in enumerate(CONDITIONS):
            ax = axes[r, c]
            ax.imshow(imgs[cond])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cond, fontsize=13)
            if c == 0:
                ax.set_ylabel(label.replace("_", " "), fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=20, help="source images per class")
    ap.add_argument("--grid-k", type=int, default=4, help="kxk grid for texture_patch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data", help="output dataset dir")
    ap.add_argument("--cache", default=".cache", help="where to download Imagenette")
    ap.add_argument("--input-dir", default=None,
                    help="use this folder of class subfolders instead of Imagenette")
    ap.add_argument("--montage-classes", type=int, default=3,
                    help="how many classes to show in figures/examples.png")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    montage_samples = []
    seen_labels = []
    counts = {c: 0 for c in CONDITIONS}

    for label, idx, rgb in iter_sources(args):
        conds = {
            "original": rgb,
            "edges": make_edges(rgb),
            "texture_patch": make_texture_patch(rgb, args.grid_k, rng),
        }
        for cond, arr in conds.items():
            save(arr, out / cond / label / f"{label}_{idx:03d}.png")
            counts[cond] += 1

        if idx == 0 and len(seen_labels) < args.montage_classes:
            montage_samples.append((label, conds))
            seen_labels.append(label)

    build_montage(montage_samples, Path("figures") / "examples.png")
    total = counts["original"]
    print(f"Wrote {total} images per condition "
          f"({', '.join(f'{k}={v}' for k, v in counts.items())}).")
    print(f"Dataset at: {out.resolve()}")
    print("Montage at: figures/examples.png")


if __name__ == "__main__":
    main()
