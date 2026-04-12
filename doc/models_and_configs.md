# Models and Configs

This document covers the two ML models that Eggsactly uses at inference time, the config files that parameterize them, and what is (and isn't) possible without retraining. For the wider pipeline context, see [architecture.md](architecture.md).

## Two models, two roles

Eggsactly runs **two separate SplineDist UNet models** sequentially:

| Model | Job | Weights | Config |
|---|---|---|---|
| **Arena detector** | Find well outlines on a downsampled full-image view | [project/models/arena_pit_v2.pth](../project/models/arena_pit_v2.pth) | [project/configs/unet_reduced_backbone_arena_wells.json](../project/configs/unet_reduced_backbone_arena_wells.json) |
| **Egg counter** | Count egg instances inside a single well sub-image | `project/models/splinedist_unet_full_400epochs_NZXT-U_2021-08-12 08-39-05.733572.pth` | [project/configs/unet_backbone_rand_zoom.json](../project/configs/unet_backbone_rand_zoom.json) |

Both models are instances of the same `SplineDist2D` architecture (defined under [project/detectors/splinedist/models/](../project/detectors/splinedist/models/)). The difference is the training data and the backbone: the arena detector uses `unet_reduced` for speed, the egg counter uses `unet_full` for accuracy.

## What SplineDist returns

Both models expose `predict_instances(image, ...)`, which returns a tuple whose second element is a dict with three keys:

| Key | Shape | Meaning |
|---|---|---|
| `points` | `(N, 2)` | Detected object centroids |
| `prob` | `(N,)` | Confidence score per detection |
| `coord` | `(N, 2, M)` | Spline control points defining the outline of each detection |

Downstream code converts the `coord` spline control points to pixel-space outlines via `get_interpolated_points()` in [project/lib/image/drawing.py](../project/lib/image/drawing.py). Circle fitting for well centers happens in [circleFinder.py:54](../project/lib/image/circleFinder.py#L54) using `fit_circle_kasa` on those outlines.

## Where models are loaded

Both models are loaded at **worker process startup**, not on demand, by `init_splinedist_network` in [project/gpu_backend/worker.py:302](../project/gpu_backend/worker.py#L302):

```python
def init_splinedist_network(type):
    wts_path = NETWORK_CONSTS[type]["wts"]
    if not Path(wts_path).is_file():
        raise FileNotFoundError(
            f"Missing GPU model weights: {wts_path}\n"
            "Trained .pth files are not in git (see .gitignore); copy them into project/models/."
        )
    networks[type] = SplineDist2D(
        Config(NETWORK_CONSTS[type]["config"], n_channel_in=3)
    )
    networks[type].cuda()
    networks[type].train(False)
    networks[type].load_state_dict(load_state_dict_compat(str(wts_path)))
```

The `NETWORK_CONSTS` dict at the top of [worker.py](../project/gpu_backend/worker.py) is where you change the path to a retrained model.

There is **also** a second load of the arena model in [circleFinder.py:22-37](../project/lib/image/circleFinder.py#L22-L37) — the `default_model` — used when the Flask app itself needs to run the model in the test path. If you change arena weights, update both locations.

## Config file reference

All three config JSONs merge with [unet_defaults.json](../project/configs/unet_defaults.json) at load time. The config loader is [project/detectors/splinedist/config.py](../project/detectors/splinedist/config.py). The interesting keys:

| Key | Default | Meaning |
|---|---|---|
| `axes` | `"YXC"` | Image axis order: Y (height), X (width), C (channels) |
| `backbone` | `"unet_full"` | `"unet_full"` or `"unet_reduced"`; reduced is ~3x faster and lower-accuracy |
| `n_control_points` | `8` | How many spline control points the model predicts per object outline. Higher = finer shape fidelity |
| `n_dim` | `2` | Always 2 (2D detection) |
| `train_patch_size` | `[160, 160]` | Crop size used during training. **Must be ≥ the size of your smallest labeled object** |
| `train_batch_size` | `4` | Training batch size (not used at inference time) |
| `train_learning_rate` | `0.0003` | Adam learning rate |
| `train_loss_weights` | `[1, 0.2]` | Weights for the two SplineDist loss terms: distance loss and probability loss |
| `zoom_min` / `zoom_max` | `null` / `null` | Zoom augmentation range; both arena and egg configs override this to `(0.9, 1.1)` |
| `focused_patch_proportion` | `null` | Arena config sets `0.85` — fraction of training crops that must contain at least one object |
| `skip_partials` | `false` | Arena config sets `true` — skip training crops where an object is only partially visible |
| `deform_sigma` | `0` | Elastic deformation sigma for augmentation |

`n_channel_in` is passed explicitly as a constructor argument (not read from JSON) and is hardcoded to `3` in [worker.py:310](../project/gpu_backend/worker.py#L310) and [circleFinder.py:33](../project/lib/image/circleFinder.py#L33). If your images are grayscale or have more channels, you'll need to change both call sites and retrain.

## No training code lives in this repo

Important to state plainly: **there is no training loop, dataset loader, augmentation pipeline, or loss-computation code in this repository.** The `.pth` files under `project/models/` were trained externally and committed (well — they used to be committed; they're now in `.gitignore`, per the recent worker.py change).

Everything under [project/detectors/splinedist/](../project/detectors/splinedist/) is **inference-only**:
- `models/model2d.py` — the `SplineDist2D` class and its forward pass
- `models/unet_block.py` — UNet backbone
- `spline_generator.py` — spline math
- `geometry/geom2d.py` — 2D geometry utilities
- `nms.py` — non-max suppression for overlapping detections

There is no `train.py`, no `dataset.py`, no `loss.py`. To retrain either model you need to use the **upstream SplineDist project** (https://github.com/uhlmanngroup/splinedist), prepare a labeled dataset in the format it expects (label images where each object has a unique integer ID), and run their training script.

## `modelRevDates.json`

[project/models/modelRevDates.json](../project/models/modelRevDates.json) tracks the release date of the **egg counter** (not the arena detector):

```json
{
    "latest": "splinedist_unet_full_400epochs_NZXT-U_2021-08-12 08-39-05.733572.pth",
    "models": {
        "splinedist_unet_full_400epochs_NZXT-U_2021-08-12 08-39-05.733572.pth": "2021-11-09"
    }
}
```

It is read by both the worker ([worker.py:82](../project/gpu_backend/worker.py#L82)) and `SessionManager` ([sessionManager.py:35-36](../project/lib/web/sessionManager.py#L35-L36)). The worker uses it to tag every prediction with the model version so results can be audited later. `SessionManager` uses it to print the model revision date as a header in exported CSVs.

If you drop a new egg model into `project/models/`, update `modelRevDates.json` to reference it (set `"latest"` to the new filename and add a matching entry under `"models"`).

## The `ARENA_IMG_RESIZE_FACTOR` constant

Hardcoded at [circleFinder.py:21](../project/lib/image/circleFinder.py#L21):

```python
ARENA_IMG_RESIZE_FACTOR = 0.186
```

This is the downscale factor applied to an input image before feeding it to the arena detector. It was chosen to match the resolution at which `arena_pit_v2.pth` was trained.

**If you retrain the arena model** on images that have a different native resolution or a different "natural" scale, you must update this constant. The worker applies it in-place when preparing the arena task ([worker.py](../project/gpu_backend/worker.py) around the arena task handler), and `CircleFinder` uses it in its `resize_image` and `resize_image_shape` helpers when running in the in-process test path.

The factor of `0.186` works out to roughly 1/5.4. A 4000×3000 pixel original becomes 744×558 at the arena model's input. If your images are significantly smaller, that downsample may leave too few pixels for wells to be detectable; if significantly larger, the factor should get smaller to stay inside the model's effective receptive field.

## Error handling at load time

The user recently added a `FileNotFoundError` guard ([worker.py:304-308](../project/gpu_backend/worker.py#L304-L308)) so that a missing `.pth` raises a clear error instead of a confusing torch serialization failure. If you are adapting this project and haven't yet copied the trained weights into `project/models/`, that is the error you will see first. It fires during worker startup, not on the first request.

## Summary: what requires retraining

| Change | Requires retraining? | Which model? |
|---|---|---|
| Different `(rows, cols)` arrangement, same well shape and image scale | **No** (usually) | — |
| Different well shape (circular → square, oval, etc.) | **Yes** | Arena detector |
| Significantly different image resolution or microscope setup | **Yes** | Arena detector (and probably egg counter) |
| Counting something other than fly eggs | **Yes** | Egg counter |
| Same objects but different background / lighting / staining | **Probably yes** | Depends which stage fails first |
| Adding extra information (e.g., a 4-channel input) | **Yes** | Both; also code changes to `n_channel_in` |

See [adapting_to_new_images.md](adapting_to_new_images.md) for how to decide which path to take and what steps are involved.
