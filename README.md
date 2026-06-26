# Shape-vs-texture control dataset

A control dataset that tests whether an image classifier relies on global object
**shape** or local **texture**, following Geirhos et al., *ImageNet-trained CNNs
are biased towards texture* (ICLR 2019,
[arXiv:1811.12231](https://arxiv.org/abs/1811.12231)).

- [`control-dataset/`](control-dataset/) — generator, evaluator, and the dataset
  (`data/original`, `data/edges`, `data/texture_patch`). See its
  [README](control-dataset/README.md) to run.
- [`report/`](report/) — the blog post ([`blog.md`](report/blog.md)).

A standard ResNet-50 scores 99.5% on the original images, 98.0% when the shape is
scrambled, and 26.0% when only the shape (edges) remains: a +72 point texture
bias.
