# Training Guide: How to Train an Egg Counter on Your Images

This is a step-by-step guide to training a SplineDist egg-counting model on your own images, from raw data to a `.pth` file you can drop into `newapp/models/` and use with the inference pipeline.

## Prerequisites

- A working newapp environment (`cd newapp && uv sync`).
- A CUDA-capable GPU with at least 4 GB of VRAM. Training *will* run on CPU but takes roughly 20x longer.
- 30–100+ images of your chambers with eggs visible.
- Patience for annotation. Labeling is the bottleneck — training itself is largely automated.

## Overview

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│ 1. Annotate │ -> │ 2. Convert   │ -> │ 3. Split     │ -> │ 4. Train    │
│    eggs in  │    │    to masks  │    │    train/val  │    │             │
│    images   │    │              │    │              │    │             │
└─────────────┘    └──────────────┘    └──────────────┘    └──────┬──────┘
                                                                  │
                                                                  v
                                                         ┌──────────────┐
                                                         │ 5. Validate  │
                                                         │    & deploy  │
                                                         └──────────────┘
```

## Step 1: Annotate your images

The egg counter is a SplineDist instance-segmentation model. It needs **per-egg polygon labels** — not bounding boxes, not class labels.

### What the labels look like

For each training image, you produce a **label image** of the same dimensions where:
- Background is 0
- Each egg is filled with a unique positive integer (1, 2, 3, ...)
- Two eggs that touch must have different IDs

### Recommended annotation tool: Labelme

[Labelme](https://github.com/wkentaro/labelme) is a lightweight polygon annotation GUI. Install it with `pip install labelme`, then:

```bash
labelme path/to/images/
```

For each image:
1. Draw a polygon around each egg. The label name doesn't matter (use "egg" for all of them) — what matters is that each polygon is a distinct object.
2. Save. Labelme writes a `.json` file alongside the image.

### Alternative: napari

If you're comfortable in Python, [napari](https://napari.org/) with a Labels layer is faster for dense annotations. Paint each egg with a distinct integer ID. Export the Labels layer as a `.png`.

### How many images do you need?

| Dataset size | Approximate accuracy | Good for |
|---|---|---|
| 20 images, ~100 eggs | Rough proof-of-concept | Checking if the approach works |
| 50 images, ~300 eggs | Decent | Most lab use cases |
| 100+ images, ~1000+ eggs | Production quality | Publication-grade counting |

**Diversity matters more than volume.** Vary the lighting, focus, background, and egg density across your training images.

### Bootstrapping with the existing model

If the upstream egg model (`splinedist_unet_full_400epochs_NZXT-U_...pth`) gets even ~50% of eggs right on your images, you can use it to pre-annotate:

1. Run it on your images with the inference pipeline.
2. Open the annotated outputs in napari or Labelme.
3. Fix the mistakes (add missed eggs, remove false detections).
4. Save as labels.

This is dramatically faster than labeling from scratch.

## Step 2: Convert annotations to masks

### From Labelme JSON

```bash
cd newapp/training
python labelme_to_masks.py path/to/labelme_jsons/ path/to/masks/
```

This converts each `.json` to a 16-bit PNG mask where each polygon gets a unique integer. The output filenames match the input stems, so `foo.json` -> `masks/foo.png`.

### From napari

If you exported label layers directly as PNGs, they're already in the right format. Just make sure they're single-channel integer images (not RGB).

### Verification

Spot-check a few with the visualizer:

```bash
python visualize.py path/to/images/ path/to/masks/ --n 5
```

Open the output PNGs in `viz_output/`. Each egg should be a distinct color with no gaps or overlaps.

## Step 3: Split into train/val

```bash
python prepare_data.py path/to/images/ path/to/masks/ my_dataset/ \
    --val-fraction 0.15
```

This creates:

```
my_dataset/
├── train/
│   ├── images/     # 85% of your data (symlinked)
│   └── labels/
└── val/
    ├── images/     # 15% held out
    └── labels/
```

Images and labels are matched by filename stem (`foo.jpg` pairs with `foo.png`).

## Step 4: Train

```bash
python train.py my_dataset/ \
    --epochs 400 \
    --batch-size 4 \
    --lr 3e-4 \
    --outdir checkpoints/
```

### What happens during training

1. **contoursize_max** is set to `n_control_points` from the config (default: 8). This ensures the training targets have the same number of channels as the model's output layer. For each foreground pixel, the target is 8 uniformly-sampled contour points of the object it belongs to (8 points x 2 coords = 16 values, matching the model's 16-channel output).
2. Random 160x160 patches are sampled from your images, biased toward patches that contain eggs (90% foreground / 10% random).
3. Each patch is augmented (random flips, rotations, brightness/contrast jitter).
4. The model predicts probability maps and spline-distance parameters for each patch.
5. The loss compares predictions against the EDT-probability and spline-distance targets.
6. Validation runs every epoch with count-level accuracy: the model's `predict_instances` is called and the predicted egg count is compared against the true count.

### Output

```
checkpoints/
├── best.pth          # best validation loss
├── epoch_0050.pth
├── epoch_0100.pth
├── ...
└── epoch_0400.pth
```

### What to watch during training

```
epoch    1/400  train_loss=0.3421  val_loss=0.3892  val_mae=4.20  lr=3.00e-04  (45.2s)
epoch    2/400  train_loss=0.2817  val_loss=0.3105  val_mae=3.51  lr=3.00e-04  (44.8s)
...
epoch  200/400  train_loss=0.0312  val_loss=0.0487  val_mae=0.35  lr=3.00e-05  (44.5s)
```

- **train_loss / val_loss**: should both decrease. If train_loss keeps dropping but val_loss plateaus or increases, you're overfitting — stop early or add more data.
- **val_mae**: mean absolute count error on validation patches. Below 1.0 is good; below 0.5 is excellent.
- **lr**: the learning rate decreases automatically when validation loss plateaus (patience=20 epochs by default).

### Common problems

| Symptom | Cause | Fix |
|---|---|---|
| Loss stays flat from the start | Labels might be all zeros (empty masks) | Run `visualize.py` to check |
| train_loss drops, val_loss doesn't | Overfitting | Add more data; reduce epochs |
| val_mae > 3 after 200 epochs | Patch size might be too small for your eggs | Increase `train_patch_size` in config |
| CUDA OOM during training | Batch size too large | Reduce `--batch-size` to 2 or 1 |
| Shape mismatch error in loss | Config mismatch between model and targets | Ensure you haven't manually set `contoursize_max` to something other than `n_control_points` |

### Tuning the config

The default config ([newapp/configs/unet_backbone_rand_zoom.json](../../configs/unet_backbone_rand_zoom.json)) is a good starting point. If you need to modify it, copy it, edit, and pass `--config your_config.json`:

- **`train_patch_size`**: default `[160, 160]`. Increase if your eggs are larger than ~80px. The patch must be big enough to contain at least one whole egg.
- **`backbone`**: `"unet_full"` (default, more accurate) or `"unet_reduced"` (3x faster inference, slightly less accurate).
- **`n_control_points`**: default 8. Increase to 12 or 16 if your eggs have complex shapes (e.g., elongated or irregular). For roughly round eggs, 8 is plenty. Note: changing this also changes `contoursize_max` (they must match), so retrain from scratch if you change it — you can't resume from a checkpoint trained with a different value.
- **`zoom_min` / `zoom_max`**: default `0.9` / `1.1`. Widens to `0.7` / `1.3` if your images have significant scale variation.

## Step 5: Validate and deploy

### Visual sanity check

```bash
python visualize.py my_dataset/val/images/ my_dataset/val/labels/ \
    --weights checkpoints/best.pth --n 10
```

This overlays ground-truth labels (colored regions) and model predictions (orange outlines) on the same image. Eyeball it — predicted outlines should align with labeled eggs.

### Deploy

Copy the best checkpoint to `newapp/models/`:

```bash
cp checkpoints/best.pth ../models/my_egg_model.pth
```

Then run the inference pipeline:

```bash
cd newapp
python main.py path/to/your/image.jpg --weights models/my_egg_model.pth
```

### Active-learning iteration

Don't try to train a perfect model in one shot:

1. Label 30–50 images.
2. Train for 200 epochs.
3. Run the partially-trained model on 30–50 new unlabeled images.
4. Correct its predictions in Labelme/napari (much faster than labeling from scratch).
5. Add the corrected labels to your dataset and re-split.
6. Resume training: `python train.py my_dataset/ --resume checkpoints/best.pth --epochs 400`
7. Repeat until `val_mae` stops improving.

This active-learning loop typically cuts total labeling effort by 3–5x.

## File reference

| File | Purpose |
|---|---|
| [labelme_to_masks.py](../labelme_to_masks.py) | Convert Labelme `.json` → instance mask PNGs |
| [prepare_data.py](../prepare_data.py) | Split images + labels into `train/` and `val/` |
| [train.py](../train.py) | Main training loop |
| [dataset.py](../dataset.py) | Patch sampling + SplineDist target computation |
| [augment.py](../augment.py) | Image + label augmentation transforms |
| [visualize.py](../visualize.py) | Overlay ground-truth labels and predictions |
| [configs/unet_backbone_rand_zoom.json](../../configs/unet_backbone_rand_zoom.json) | Default model config |
| [configs/unet_defaults.json](../../configs/unet_defaults.json) | Config defaults (merged at load time) |

## FAQ

**Q: Do I need the arena model too?**
No. The arena model was only needed in the upstream web app to find well locations automatically. In newapp, you supply bounding boxes yourself via `detect_regions()` in [eggcount/chamber_detection.py](../../eggcount/chamber_detection.py). The only model you train is the egg counter.

**Q: Can I fine-tune the upstream model instead of training from scratch?**
Yes, and this is **recommended as the default approach** — pass `--resume path/to/upstream_model.pth` to `train.py`. The upstream model was trained on *Drosophila* eggs, so if yours look similar, fine-tuning converges much faster than starting from random weights. Fine-tuning is also more forgiving of any subtle differences in how the training targets are represented.

**Q: My eggs are smaller/larger than the upstream ones. What do I change?**
Adjust `train_patch_size` in your config. The patch must be large enough to contain at least one egg with some margin. For very small eggs (< 20px), the default 160x160 is fine. For eggs > 80px, increase to 256x256 or larger — but you may need to reduce batch size to avoid OOM.

**Q: Can I train on grayscale images?**
Yes — pass `n_channel_in=1` when creating the Config (edit `train.py` line where `Config(...)` is called). Make sure your inference pipeline also uses `n_channel_in=1` in `EggModel`.

**Q: How long does training take?**
On an NVIDIA RTX 3080: roughly 30–60 seconds per epoch with the default settings (100 steps, batch size 4, 160x160 patches). A full 400-epoch run is 3–7 hours. On CPU, multiply by ~20x.
