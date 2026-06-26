# A shape-vs-texture control dataset: what does an image classifier actually look at?

A ResNet-50 that scores 99.5% on a set of object photos still scores 98% after you cut each photo into sixteen tiles and shuffle them, destroying the object's shape. Cut the texture instead, leaving only the outline, and the same network collapses to 26%. The network is reading texture, not shape. This post describes a small control dataset built to make that property visible and measurable, and shows it doing so.

## The property and why it's worth testing

The usual story about convolutional networks is that they build objects out of parts: edges into contours, contours into wheels and windows, wheels and windows into a car. Geirhos et al. ([arXiv:1811.12231](https://arxiv.org/abs/1811.12231)) call this the *shape hypothesis* and then take it apart. Across nine psychophysical experiments (48,560 trials, 97 observers) they show that ImageNet-trained CNNs decide mostly on *texture*: a cat-shaped image painted with elephant skin is an elephant to a CNN and a cat to a human. Humans pick the shape category 95.9% of the time; a standard ResNet-50 picks it 22% of the time.

That gap matters for two reasons the paper makes concrete. CNNs are used as models of human vision, so a network that classifies on a cue humans largely ignore is a poor model of the thing it claims to explain. And texture reliance predicts fragility: textures wash out under noise, blur, rain and snow, while an object's outline survives. The paper's shape-trained network (their Stylized-ImageNet model) picks up robustness to distortions it never saw in training, which is what you'd expect if it had stopped leaning on a cue that noise destroys.

So the property to test is single and concrete: *given an image, does a classifier's decision follow the shape or the texture?* A good control dataset should answer that for any classifier you hand it, without retraining and without running a human study.

## The dataset

The dataset isolates the cue by removing one at a time. Each source object image becomes three aligned 224×224 versions:

- `original` — both cues present. The baseline.
- `edges` — shape kept, texture removed. A Canny edge map of the greyscaled, blurred image: black contours on white. This is the paper's own "edges" condition (their Appendix A.6).
- `texture_patch` — texture kept, shape destroyed. The image is cut into a 4×4 grid and the sixteen tiles are randomly permuted. Local texture statistics survive inside each tile; the global outline is gone. This is the scrambled-shape manipulation the paper cites (Gatys et al. 2017; Brendel & Bethge 2019).

![Three source classes shown as original, edges, and texture_patch](https://raw.githubusercontent.com/nikshithmenta/Control_Dataset/main/report/examples.png)

*Three rows of source classes (tench, English springer, cassette player), each shown as `original`, `edges`, and `texture_patch`. The edge image keeps the contour and drops the texture; the patch image keeps the texture and drops the contour.*

The point of the design is what stays fixed. Class, source image, and image size are identical across the three conditions; the only thing that varies is which cue is available. That's what separates a control dataset from a benchmark. If a model's accuracy changes between conditions, the change can't be blamed on different objects or harder images. It can only come from which cue the model was using. The headline measurement is one number:

```
gap = acc(texture_patch) - acc(edges)
```

A large positive gap means the model reads texture; a large negative gap means it reads shape; near zero means it uses both or neither. This is the same shape-versus-texture contrast as the paper's cue-conflict experiment, reduced to something you can compute on any pretrained model in a minute.

## How the data is generated

Source images come from Imagenette, a ten-class subset of ImageNet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute), pulled through `torchvision`. Using ImageNet classes is deliberate: a stock ImageNet-trained ResNet-50 can be scored on them directly, so its predictions are interpretable without any fine-tuning. Each image is center-cropped square and resized to 224×224, then the two manipulations are applied.

The `edges` condition reproduces the paper's MATLAB recipe (`1 - edge(imgaussfilt(rgb2gray(img), 2), 'Canny')`) with OpenCV:

```python
def make_edges(rgb):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grey, ksize=(0, 0), sigmaX=2.0)
    edges = cv2.Canny(blurred, 50, 150)   # white edges on black
    inverted = 255 - edges                 # black edges on white
    return np.repeat(inverted[:, :, None], 3, axis=2)
```

The `texture_patch` condition slices the image into a k×k grid and permutes the tiles, reshuffling until at least one tile has moved so the identity permutation never sneaks through:

```python
def make_texture_patch(rgb, k, rng):
    h, w, _ = rgb.shape
    ph, pw = h // k, w // k
    blocks = [rgb[i*ph:(i+1)*ph, j*pw:(j+1)*pw].copy()
              for i in range(k) for j in range(k)]
    order = list(range(len(blocks)))
    while True:
        rng.shuffle(order)
        if any(order[i] != i for i in range(len(order))):
            break
    blocks = [blocks[o] for o in order]
    rows = [np.concatenate(blocks[r*k:(r+1)*k], axis=1) for r in range(k)]
    return np.concatenate(rows, axis=0)
```

A fixed random seed makes the whole dataset reproducible. The generator writes `data/{original,edges,texture_patch}/<class>/` as PNGs and saves the montage above. The default run produces 20 images per class, 200 per condition. The full generator and evaluator are two short scripts; both are in the repository linked at the end.

## Does the dataset test what it claims?

It does, and the effect is large. A standard ImageNet-trained ResNet-50 (`torchvision` `IMAGENET1K_V2`), scored as a 10-way Imagenette classifier:

| condition | top-1 accuracy |
|---|:--:|
| `original` | 99.5% |
| `edges` | 26.0% |
| `texture_patch` | 98.0% |
| **texture − shape gap** | **+72.0 pts** |

Scrambling the global shape drops accuracy by 1.5 points. Removing texture and keeping only the outline drops it by 73.5. The model holds up when the object is in pieces and falls apart when it's only a contour, which is the texture bias Geirhos et al. report, reproduced here on a dataset you can regenerate in a few minutes. The 26% edge accuracy also matches their observation that CNNs cope badly with texture-free images partly because of the domain shift from photos to line drawings; that confound is real, which is why the dataset's decisive comparison is `texture_patch` against `edges` rather than either against `original`.

Hand the same dataset a shape-biased model and the gap should shrink or flip. That's the use of a control: the number it produces is a property of the model under test, not of the images.

## Links

- Dataset and code: https://github.com/nikshithmenta/Control_Dataset
- Paper: Geirhos et al., *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness*, ICLR 2019, https://arxiv.org/abs/1811.12231
- Imagenette: https://github.com/fastai/imagenette

## References

1. R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, W. Brendel. *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness*. ICLR 2019. arXiv:1811.12231.
2. L. A. Gatys, A. S. Ecker, M. Bethge. *Texture and art with deep neural networks*. Current Opinion in Neurobiology, 46:178–186, 2017.
3. W. Brendel, M. Bethge. *Approximating CNNs with bag-of-local-features models works surprisingly well on ImageNet*. ICLR 2019.
