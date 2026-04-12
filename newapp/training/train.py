#!/usr/bin/env python3
"""Train a SplineDist egg-counting model.

Usage
-----
    python train.py path/to/dataset/ --epochs 400 --outdir checkpoints/

where ``path/to/dataset/`` has the structure created by ``prepare_data.py``::

    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/

Saves checkpoints to ``--outdir`` and prints training/validation loss and
mean absolute count error each epoch. The best-validation checkpoint is
saved as ``best.pth``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure eggcount is importable
_NEWAPP_ROOT = Path(__file__).resolve().parent.parent
if str(_NEWAPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEWAPP_ROOT))

from eggcount.splinedist.config import Config
from eggcount.splinedist.models.model2d import SplineDist2D

from augment import default_augmenter
from dataset import SplineDistPatchDataset, load_image_label_pairs


def train_one_epoch(
    model: SplineDist2D,
    dataset: SplineDistPatchDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    steps: int,
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train(True)
    total_loss = 0.0
    for step in range(steps):
        patches, prob, dist, counts = dataset.sample_batch(batch_size)

        patches_t = torch.from_numpy(patches).float().to(device)
        patches_t = patches_t.permute(0, 3, 1, 2)  # (B,C,H,W)
        true_prob = torch.from_numpy(prob).float().to(device)
        true_dist = torch.from_numpy(dist).float().to(device)

        optimizer.zero_grad()
        result = model(patches_t)
        pred_prob = result[0].permute(0, 2, 3, 1)  # (B,H',W',1)
        pred_dist = result[1].permute(0, 2, 3, 1)  # (B,H',W',n_params)

        loss = model.loss([true_prob, true_dist], [pred_prob, pred_dist])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / steps


@torch.no_grad()
def validate(
    model: SplineDist2D,
    dataset: SplineDistPatchDataset,
    device: torch.device,
    batch_size: int,
    steps: int,
) -> tuple[float, float]:
    """Run validation and return ``(mean_loss, mean_abs_count_error)``."""
    model.train(False)
    total_loss = 0.0
    abs_errors = []

    for step in range(steps):
        patches, prob, dist, counts = dataset.sample_batch(batch_size)

        patches_t = torch.from_numpy(patches).float().to(device)
        patches_t = patches_t.permute(0, 3, 1, 2)
        true_prob = torch.from_numpy(prob).float().to(device)
        true_dist = torch.from_numpy(dist).float().to(device)

        result = model(patches_t)
        pred_prob = result[0].permute(0, 2, 3, 1)
        pred_dist = result[1].permute(0, 2, 3, 1)

        loss = model.loss([true_prob, true_dist], [pred_prob, pred_dist])
        total_loss += loss.item()

        # Count accuracy: run predict_instances on each patch
        for i in range(patches_t.shape[0]):
            single = patches_t[i].permute(1, 2, 0).cpu().numpy()
            _, pred = model.predict_instances(single)
            pred_count = len(pred["points"])
            abs_errors.append(abs(int(counts[i]) - pred_count))

    mean_loss = total_loss / steps
    mean_abs_err = float(np.mean(abs_errors)) if abs_errors else 0.0
    return mean_loss, mean_abs_err


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Dataset root (train/ + val/)")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=100,
                        help="Gradient steps per epoch")
    parser.add_argument("--val-steps", type=int, default=20,
                        help="Validation batches per epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--config", type=str, default="unet_backbone_rand_zoom.json",
                        help="Config JSON (bare filename under newapp/configs/)")
    parser.add_argument("--outdir", type=Path, default=Path("checkpoints"),
                        help="Directory for saved checkpoints")
    parser.add_argument("--device", type=str, default=None,
                        choices=("cuda", "cpu"))
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from a checkpoint .pth")
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")

    # ---- Load data ----
    train_dir = args.dataset_dir / "train"
    val_dir = args.dataset_dir / "val"
    print("loading training data...")
    X_trn, Y_trn = load_image_label_pairs(
        train_dir / "images", train_dir / "labels"
    )
    print("loading validation data...")
    X_val, Y_val = load_image_label_pairs(
        val_dir / "images", val_dir / "labels"
    )

    # ---- Determine contoursize_max ----
    # contoursize_max must equal n_control_points (from the config) so that
    # the training targets have the same number of channels as the model's
    # output layer. The model predicts n_params = 2 * n_control_points values
    # per pixel, so spline_dist must sample exactly n_control_points contour
    # points to produce matching 2 * n_control_points target channels.
    #
    # Note: get_contoursize_max(Y_trn) returns the raw contour length in
    # pixels (often hundreds) — that is NOT the right value here.
    _temp_cfg = Config(args.config, n_channel_in=3, contoursize_max=1)
    n_control_points = _temp_cfg.n_params // 2  # n_params = 2 * n_control_points
    contoursize_max = n_control_points
    print(f"n_control_points = {n_control_points}, contoursize_max = {contoursize_max}")

    # ---- Build model ----
    config = Config(args.config, n_channel_in=3, contoursize_max=contoursize_max)
    model = SplineDist2D(config, train=True)
    if device.type == "cuda":
        model.cuda()
    else:
        model.to(device)

    if args.resume:
        from eggcount.torch_utils import load_state_dict_compat
        state = load_state_dict_compat(str(args.resume), map_location=device)
        model.load_state_dict(state)
        print(f"resumed from {args.resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_reduct_factor or 0.5,
        patience=config.lr_patience or 20,
        verbose=True,
    )

    # ---- Build datasets ----
    patch_size = tuple(config.train_patch_size)
    grid = tuple(config.grid)

    train_dataset = SplineDistPatchDataset(
        X_trn, Y_trn,
        patch_size=patch_size,
        contoursize_max=contoursize_max,
        grid=grid,
        foreground_prob=config.train_foreground_only or 0.9,
        augmenter=default_augmenter,
    )
    val_dataset = SplineDistPatchDataset(
        X_val, Y_val,
        patch_size=patch_size,
        contoursize_max=contoursize_max,
        grid=grid,
        foreground_prob=1.0,  # Always center on objects for consistent eval
        augmenter=None,
    )

    # ---- Train ----
    args.outdir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_dataset, optimizer, device,
            args.batch_size, args.train_steps,
        )
        val_loss, val_mae = validate(
            model, val_dataset, device,
            args.batch_size, args.val_steps,
        )
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:4d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_mae={val_mae:.2f}  "
            f"lr={lr:.2e}  "
            f"({elapsed:.1f}s)"
        )

        # Save periodic checkpoints
        if epoch % 50 == 0 or epoch == args.epochs:
            ckpt_path = args.outdir / f"epoch_{epoch:04d}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved {ckpt_path}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = args.outdir / "best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  new best val_loss={val_loss:.4f} -> {best_path}")

    print(f"\ndone. best checkpoint: {args.outdir / 'best.pth'}")
    print(
        f"copy it to newapp/models/ and run:\n"
        f"  python newapp/main.py your_image.jpg --weights {args.outdir / 'best.pth'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
