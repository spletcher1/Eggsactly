# Training a Custom SplineDist UNet Model with Eggsactly

This guide walks through the complete process of training a new SplineDist UNet model
on your own images using the neural network infrastructure already built into this project.

---

## Do My Images Need to Contain Eggs?

**Yes — your training images must contain the objects you want to detect** (eggs, or
whatever you are counting). The model learns the appearance and shape of objects by
seeing labeled examples of them.

- **Most images** (~25 out of 30) should contain objects with instance masks labeling
  each one individually.
- **A few empty images** (0–5) can be included and will be used as negative examples,
  but they contribute very little to learning — the model spends 90% of training drawing
  patches that contain at least one object (controlled by `train_foreground_only: 0.9`
  in the config).
- Aim for a **range of densities**: some images with few eggs and some with many. The
  existing egg model was trained on images with counts ranging from 0 to ~75 per
  agarose region.

---

## Prerequisites

### Install dependencies

The project's `pyproject.toml` / `uv.lock` covers most packages. The critical
non-obvious ones for the SplineDist training stack are:

```bash
# Inside the project virtualenv:
pip install stardist==0.9.2        # provides the C-extension NMS used at inference time
pip install csbdeep==0.8.2         # normalisation utilities and tile iterators
pip install elasticdeform           # elastic grid deformation for augmentation
pip install edt                     # fast Euclidean distance transform (optional but faster)
pip install numexpr                 # fast normalisation arithmetic (optional but faster)
# PyTorch with CUDA must be installed separately — see https://pytorch.org/get-started
```

### Hardware

A CUDA-capable GPU is strongly recommended. The project auto-detects one via:

```python
# project/detectors/splinedist/constants.py
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Training 400 epochs on 25 images took roughly a day on the NZXT-U machine used to
produce the bundled egg model.

---

## Step 1 — Prepare Your Images

### Image format

Each image must be a `float32` NumPy array with shape `(H, W, 3)` — RGB channels last
(axes `YXC`). Do not normalise to [0, 1] yourself; the model calls `csbdeep.utils.normalize`
internally during inference. For training, values in the uint8 range 0–255 are fine.

```python
import cv2, numpy as np

def load_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)
```

### Mask format

Each mask must be a 2D integer NumPy array with shape `(H, W)` matching its image:

- `0` = background
- Every positive integer = one unique object instance

Each individual egg (or object) must have its own unique label. Two eggs that happen to
touch each other must still have **different** integer values — this is instance
segmentation, not semantic segmentation.

```
Example 3×3 mask:
  0  0  0
  0  1  0    ← object 1
  0  0  2    ← object 2 (different egg, different integer)
```

### Labelling tools

Any tool that exports instance segmentation masks will work:

| Tool | Notes |
|---|---|
| [Napari](https://napari.org/) + `napari-annotator` | Python-native, good for bulk labelling |
| [QuPath](https://qupath.github.io/) | Excellent for biological images; export as labelled TIF |
| [labelme](https://github.com/labelmeai/labelme) | JSON polygons → convert to mask (see below) |
| FIJI/ImageJ | Use the ROI Manager; export via `Image > Overlay > To ROI Manager` → save as labelled image |

#### Converting labelme polygon JSON to a mask

```python
import json, numpy as np
from skimage.draw import polygon

def labelme_json_to_mask(json_path: str, height: int, width: int) -> np.ndarray:
    with open(json_path) as f:
        data = json.load(f)
    mask = np.zeros((height, width), dtype=np.int32)
    for idx, shape in enumerate(data["shapes"], start=1):
        pts = np.array(shape["points"])  # shape (N, 2), col-first (x, y)
        rr, cc = polygon(pts[:, 1], pts[:, 0], shape=(height, width))
        mask[rr, cc] = idx
    return mask
```

---

## Step 2 — Split Into Train and Validation Sets

With 30 images a reasonable split is **25 train / 5 validation**.

```python
import os, random, numpy as np

# --- Load all image/mask pairs ---
image_dir = "my_data/images"
mask_dir  = "my_data/masks"

filenames = sorted(os.listdir(image_dir))
random.seed(42)
random.shuffle(filenames)

X_all = [load_image(os.path.join(image_dir, f)) for f in filenames]
Y_all = [np.load(os.path.join(mask_dir, f.replace(".jpg", ".npy"))) for f in filenames]

split = 25
X_train, Y_train = X_all[:split], Y_all[:split]
X_val,   Y_val   = X_all[split:], Y_all[split:]
```

> **Tip:** Make sure the validation set contains a representative mixture of egg
> densities (not just the easy empty images).

---

## Step 3 — Choose a Config

Two ready-made configs are relevant for egg-style detection. Both live in
`project/configs/`. The `Config` class reads these at runtime; its path argument is
**relative to the repository root** (`Eggsactly/`).

### `unet_backbone_rand_zoom.json` — used for the bundled egg model

```json
{
    "backbone": "unet_full",
    "train_epochs": 400,
    "train_steps_per_epoch": 100,
    "validation_batch_size": 4,
    "validation_steps_per_epoch": 100,
    "zoom_min": 0.9,
    "zoom_max": 1.1
}
```

Unspecified parameters fall back to `project/configs/unet_defaults.json`:

| Parameter | Default | What it controls |
|---|---|---|
| `train_patch_size` | `[160, 160]` | Size of random crops fed to the network. Increase if your eggs are large relative to the image. |
| `train_batch_size` | `4` | Patches per gradient step. Reduce if you run out of GPU memory. |
| `train_learning_rate` | `0.0003` | Adam learning rate. |
| `n_control_points` | `8` | Number of spline control points describing each egg contour. 8 is sufficient for smooth ellipse-like shapes. |
| `train_loss_weights` | `[1, 0.2]` | Weight of probability loss vs. contour distance loss. |
| `train_foreground_only` | `0.9` | Fraction of patches guaranteed to contain at least one object. |
| `lr_reduct_factor` | `0.5` | Multiply learning rate by this when validation loss plateaus. |
| `lr_patience` | `20` | Epochs without improvement before reducing LR. |
| `grid_subsampling_factor` | `[2, 2]` | Output is at half the input resolution. Keep at `[2, 2]` unless eggs are very small. |

### `unet_reduced_backbone_arena_wells.json` — lighter alternative

Uses `backbone: "unet_reduced"` — a smaller 3-level U-Net. Trains faster and uses less
memory, at the cost of some accuracy. Useful if you have limited GPU VRAM or your
objects have simpler shapes.

### Choosing a backbone

| Backbone | Depth | When to use |
|---|---|---|
| `unet_full` | 5-level U-Net | Default. Best accuracy. Requires ≥ 8 GB VRAM. |
| `unet_reduced` | 3-level U-Net | Faster, lower VRAM. Good first attempt if `unet_full` is slow or OOM. |
| `fcrn_a` | FCN variant | Experimental. Not used by the production models here. |

---

## Step 4 — Write a Training Script

The training orchestrator is the `Looper` class in
`project/detectors/splinedist/looper.py`. Create a script at the **repository root**
(e.g. `train_my_model.py`) so that relative config paths resolve correctly.

```python
# train_my_model.py
# Run from the Eggsactly/ repository root:
#   python train_my_model.py

import sys, os
sys.path.insert(0, os.path.dirname(__file__))  # ensure project/ is importable

import torch
import numpy as np

from project.detectors.splinedist.config import Config
from project.detectors.splinedist.models.model2d import SplineDist2D
from project.detectors.splinedist.looper import Looper
from project.detectors.splinedist.constants import DEVICE

# ── 1. Load data ─────────────────────────────────────────────────────────────
# Replace these with your actual loading code (see Steps 1 & 2).
X_train: list  # list of float32 np.ndarray, each (H, W, 3)
Y_train: list  # list of int    np.ndarray, each (H, W) — instance labels
X_val:   list
Y_val:   list

# ── 2. Config ─────────────────────────────────────────────────────────────────
# Path is relative to the Eggsactly/ root.
config = Config("project/configs/unet_backbone_rand_zoom.json")

# ── 3. Model ──────────────────────────────────────────────────────────────────
model = SplineDist2D(config, train=True).to(DEVICE)

# ── 4. Optimizer ──────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=config.train_learning_rate)

# ── 5. Learning-rate scheduler (mirrors the lr_patience logic in the config) ──
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=config.lr_reduct_factor,
    patience=config.lr_patience,
    verbose=True,
)

# ── 6. Build train and validation Loopers ─────────────────────────────────────
#   augmenter=None  →  uses the identity function; augmentation is applied
#   inside SplineDistData2D via rotation.py regardless.
train_looper = Looper(
    network=model,
    config=config,
    device=DEVICE,
    loss=model.loss,      # stored but training uses model.loss internally
    optimizer=optimizer,
    augmenter=None,
    X=X_train,
    Y=Y_train,
    validation=False,
)

val_looper = Looper(
    network=model,
    config=config,
    device=DEVICE,
    loss=model.loss,
    optimizer=optimizer,
    augmenter=None,
    X=X_val,
    Y=Y_val,
    validation=True,
)

# ── 7. Training loop ──────────────────────────────────────────────────────────
save_path = "my_splinedist_model.pth"
best_val_mae = float("inf")

for epoch in range(1, config.train_epochs + 1):
    print(f"\n=== Epoch {epoch}/{config.train_epochs} ===")
    train_looper.run(epoch)
    val_mae = val_looper.run(epoch)

    # Step the LR scheduler on validation MAE
    scheduler.step(val_mae)

    # Save the best model
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), save_path)
        print(f"  → Saved new best model (val MAE={val_mae:.3f}) to {save_path}")
```

---

## Step 5 — Monitor Training

The `Looper.log()` method prints after every epoch:

```
Train:
    Average loss:  0.0842
    Mean error:    0.123
    Mean absolute error: 0.887
    Error deviation: 1.204
```

- **Average loss** — combined probability + contour loss. Should decrease over epochs.
- **Mean error** — signed difference (true count − predicted count). Should be near 0.
- **Mean absolute error (MAE)** — unsigned count error per patch. The main metric to
  watch; the bundled egg model achieves < 1 egg MAE on held-out data.

Optional: pass `matplotlib.axes.Axes` objects to the Looper constructors as the `plots`
argument to see live scatter plots and loss curves during training.

### Expected training trajectory with 30 images

| Epochs | Expected MAE | Notes |
|---|---|---|
| 0–50 | 5–15 | Model is still learning basic blob detection |
| 50–150 | 2–5 | Contour shapes start to converge |
| 150–300 | 1–3 | Main learning phase |
| 300–400 | 0.5–2 | Refinement; LR may have stepped down once or twice |

With only 30 images, some overfitting is possible. Watch that training MAE and
validation MAE stay within ~0.5 of each other by epoch 300.

---

## Step 6 — Run Inference with Your Trained Model

```python
import torch, numpy as np
from project.detectors.splinedist.config import Config
from project.detectors.splinedist.models.model2d import SplineDist2D
from project.detectors.splinedist.constants import DEVICE

config = Config("project/configs/unet_backbone_rand_zoom.json")
model  = SplineDist2D(config, train=False).to(DEVICE)
model.load_state_dict(torch.load("my_splinedist_model.pth", map_location=DEVICE))
model.eval()

# image must be float32 np.ndarray (H, W, 3)
image = load_image("path/to/new_image.jpg")

labels, details = model.predict_instances(
    image,
    prob_thresh=0.5,   # lower → detect more objects (more false positives)
    nms_thresh=0.4,    # lower → suppress more overlapping detections
)

egg_count = len(details["points"])
print(f"Detected {egg_count} objects")

# labels  — (H, W) integer array; each unique nonzero value is one detected object
# details — dict with keys: "points" (centroids), "coord" (contour polygons),
#            "prob" (confidence scores)
```

---

## Step 7 — Integrate into the Eggsactly Web App (Optional)

The GPU worker in `project/gpu_backend/worker.py` loads models by task type. To use
your model for egg counting, update the model path in
`project/gpu_backend/worker.py` where `GPUTaskTypes.egg` is handled, pointing it at
your new `.pth` file and its corresponding config JSON.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch or patch size too large | Reduce `train_batch_size` to 2, or `train_patch_size` to `[128, 128]` |
| MAE never drops below 5 | Masks have labelling errors | Inspect a few masks visually; check that touching objects have different integer labels |
| Validation MAE diverges strongly from training MAE | Overfitting | Enable zoom augmentation (`zoom_min: 0.8`, `zoom_max: 1.2`); increase `deform_sigma` in config |
| `patch_size ... larger than data shape` error | An image is smaller than the patch size | Either resize that image or reduce `train_patch_size` |
| `stardist` import error during inference | NMS C extension missing | Reinstall `stardist==0.9.2` and ensure it compiled successfully |
| Very slow training, GPU idle | Data loading bottleneck | Pre-compute masks once before training rather than reloading from disk each epoch |

---

## File Reference

| File | Purpose |
|---|---|
| `project/detectors/splinedist/looper.py` | `Looper` — training + validation loop |
| `project/detectors/splinedist/models/model2d.py` | `SplineDist2D` — full model, loss, and inference |
| `project/detectors/splinedist/models/database.py` | `SplineDistData2D` — dynamic data generator with augmentation |
| `project/detectors/splinedist/config.py` | `Config` — loads JSON config with defaults fallback |
| `project/detectors/splinedist/rotation.py` | `RotationHelper` — random rotation + elastic deformation |
| `project/configs/unet_defaults.json` | Default hyperparameters (all fields) |
| `project/configs/unet_backbone_rand_zoom.json` | Config used to train the bundled egg model |
| `project/configs/unet_reduced_backbone_arena_wells.json` | Lighter config for the arena/well-finder model |
| `project/detectors/splinedist/constants/phi_8.npy` | Pre-computed B3 spline basis matrix (M=8 control points) |
| `project/models/*.pth` | Pre-trained weight files for reference |
