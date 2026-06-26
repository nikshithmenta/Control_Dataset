"""Evaluate a pretrained ResNet-50 on the shape-vs-texture control dataset.

For each condition (original / edges / texture_patch) the model classifies every
image and we report top-1 accuracy over the 10 Imagenette classes. The model only
ever sees ImageNet training images, so the difference between conditions reflects
which cue it learned to rely on.

Prediction is restricted to the 10 ImageNet logits that correspond to the
Imagenette classes (standard 10-way Imagenette evaluation). The headline number
is the texture-vs-shape gap:

    gap = acc(texture_patch) - acc(edges)

A positive gap means the model does better when only texture survives than when
only shape survives: a texture bias, the paper's central finding.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

CONDITIONS = ("original", "edges", "texture_patch")

# Imagenette label -> ImageNet1k class index (see ResNet50_Weights.meta).
LABEL_TO_IMAGENET_IDX = {
    "tench": 0,
    "english_springer": 217,
    "cassette_player": 482,
    "chain_saw": 491,
    "church": 497,
    "french_horn": 566,
    "garbage_truck": 569,
    "gas_pump": 571,
    "golf_ball": 574,
    "parachute": 701,
}


@torch.no_grad()
def accuracy_for_condition(model, preprocess, cond_dir: Path, allowed_idx, idx_to_label, device):
    correct = total = 0
    allowed = torch.tensor(allowed_idx, device=device)
    for class_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
        true_label = class_dir.name
        for fp in sorted(class_dir.glob("*.png")):
            x = preprocess(Image.open(fp).convert("RGB")).unsqueeze(0).to(device)
            logits = model(x)[0]
            restricted = logits[allowed]                 # only the 10 classes
            pred_idx = allowed[int(restricted.argmax())].item()
            pred_label = idx_to_label[pred_idx]
            correct += int(pred_label == true_label)
            total += 1
    return correct, total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="dataset dir produced by generate.py")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).eval().to(device)
    preprocess = weights.transforms()

    allowed_idx = sorted(LABEL_TO_IMAGENET_IDX.values())
    idx_to_label = {v: k for k, v in LABEL_TO_IMAGENET_IDX.items()}

    data = Path(args.data)
    results = {}
    print(f"device={device}  model=ResNet-50 (IMAGENET1K_V2)\n")
    for cond in CONDITIONS:
        c, t = accuracy_for_condition(model, preprocess, data / cond,
                                      allowed_idx, idx_to_label, device)
        acc = c / t if t else 0.0
        results[cond] = {"correct": c, "total": t, "accuracy": acc}
        print(f"{cond:<14} {acc*100:6.2f}%  ({c}/{t})")

    gap = results["texture_patch"]["accuracy"] - results["edges"]["accuracy"]
    results["texture_minus_shape_gap"] = gap
    print(f"\ntexture - shape gap: {gap*100:+.2f} pts "
          f"(positive => texture bias)")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
