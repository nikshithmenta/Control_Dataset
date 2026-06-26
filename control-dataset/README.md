# Shape-vs-texture control dataset

A small control dataset that tests one property of an image classifier: does it
rely on global object **shape** or on local **texture**? The design follows
Geirhos et al., *ImageNet-trained CNNs are biased towards texture; increasing
shape bias improves accuracy and robustness*, ICLR 2019
([arXiv:1811.12231](https://arxiv.org/abs/1811.12231)).

Each source object image is turned into three aligned 224×224 conditions:

| condition       | shape | texture | how |
|-----------------|:-----:|:-------:|-----|
| `original`      | ✓ | ✓ | baseline |
| `edges`         | ✓ | ✗ | Canny edge map of a blurred greyscale image |
| `texture_patch` | ✗ | ✓ | 4×4 grid of the image, patches randomly shuffled |

Only the available cue changes between conditions; class, source image and size
are held fixed. So a difference in a model's accuracy across conditions is
attributable to cue reliance alone, which is what makes this a control rather
than a benchmark.

## Run

```bash
pip install -r requirements.txt
python generate.py            # downloads Imagenette, writes data/ + figures/examples.png
python evaluate.py            # runs ResNet-50, writes results.json
```

Generation downloads Imagenette (10 ImageNet classes) on first run. To use your
own images instead:

```bash
python generate.py --input-dir my_images   # my_images/<class>/*.jpg
```

Useful flags: `--per-class N` (images per class), `--grid-k K` (patch grid),
`--seed S`.

## Result

A standard ImageNet-trained ResNet-50 (`torchvision` `IMAGENET1K_V2`) on 20
images/class:

| condition       | top-1 accuracy |
|-----------------|:--------------:|
| `original`      | 99.5% |
| `edges`         | 26.0% |
| `texture_patch` | 98.0% |

texture − shape gap = **+72 points**. Destroying the global shape barely affects
the model; removing texture nearly breaks it. That is the texture bias the paper
describes.

## Files

- `generate.py` — builds the dataset and `figures/examples.png`.
- `evaluate.py` — scores ResNet-50 per condition, writes `results.json`.
- `data/` — the control dataset (`original/`, `edges/`, `texture_patch/`, each
  with one subfolder per class).

## Links

- Dataset + code: https://github.com/nikshithmenta/Control_Dataset
- Paper: https://arxiv.org/abs/1811.12231
- Imagenette: https://github.com/fastai/imagenette
