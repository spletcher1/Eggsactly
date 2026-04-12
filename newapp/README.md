# newapp — Local Egg Counter

A stripped-down, single-machine version of Eggsactly. No Flask, no SocketIO, no SQL database, no auth, no separate GPU worker process. Just:

1. Load an image.
2. Ask a small function `detect_regions(img)` where the egg-laying surfaces are.
3. Crop each region and run the trained SplineDist egg counter on it.
4. Write an annotated image, a JSON, and a CSV alongside the input.

If you want a GUI, it's a thin wrapper around [`eggcount.pipeline.analyze_image`](eggcount/pipeline.py) — add a PyQt / tkinter / whatever front-end later; none of the core logic changes.

## Directory layout

```
newapp/
├── main.py                      # CLI entry point
├── configs/                     # SplineDist config JSONs (copied from upstream)
│   ├── unet_defaults.json
│   └── unet_backbone_rand_zoom.json
├── models/                      # DROP YOUR .pth WEIGHT FILES HERE
└── eggcount/
    ├── chamber_detection.py     # ← PLUG-IN POINT: define your regions here
    ├── egg_model.py             # Thin wrapper over SplineDist inference
    ├── pipeline.py              # End-to-end analyze_image() + rendering
    ├── drawing.py               # Spline → pixel outline conversion
    ├── torch_utils.py           # Torch weights loader helper
    └── splinedist/              # Copied from upstream, inference-only
```

## Quick start

1. **Copy a trained egg-counter `.pth`** into `newapp/models/`. The upstream project ships one at `project/models/splinedist_unet_full_400epochs_NZXT-U_2021-08-12 08-39-05.733572.pth`. Any SplineDist 2D checkpoint compatible with [`configs/unet_backbone_rand_zoom.json`](configs/unet_backbone_rand_zoom.json) will work.

2. **Install dependencies** (from the repo root, in your existing venv):
   ```
   pip install torch numpy opencv-python scipy scikit-image csbdeep tqdm Pillow
   ```
   GPU inference needs a CUDA-capable torch build; CPU will work but is slow.

3. **Run the CLI**:
   ```
   python newapp/main.py path/to/your/image.jpg
   ```
   Outputs land next to the input (or pass `--outdir`):
   - `<stem>_annotated.png`
   - `<stem>_counts.json`
   - `<stem>_counts.csv`

4. **Plug in your own chamber detection.** Open [`eggcount/chamber_detection.py`](eggcount/chamber_detection.py) and replace `detect_regions` with your own logic. The default returns a single region covering the entire image, which is only useful as a smoke test.

## The plug-in point: `detect_regions`

Everything that made the upstream project complicated — the arena-detection neural network, the grid-recovery histogram binning, the chamber-type enum, the coupled per-type `getSubImageBBoxes` branches — has been deleted and replaced with a single function:

```python
def detect_regions(img: np.ndarray) -> list[Region]:
    """Return the egg-laying regions in img."""
    ...
```

A `Region` carries a bounding box `(x, y, w, h)` and, optionally, a circle `(cx, cy, r)`. If the circle is set, pixels outside it are masked out of the crop before the egg counter runs — handy when your chambers are round pits and you don't want the counter to see eggs belonging to the pit next door.

You can:

- **Hardcode it.** For quick testing, return bboxes you measured off a reference image. See the docstring of `detect_regions` for two example patterns (single centered circle; fixed 3×3 grid).
- **Compute it with OpenCV.** Hough circles, template matching, contour detection on a binarized image, etc.
- **Plug in a small ML model.** Train your own lightweight detector and call it from inside `detect_regions`. The pipeline doesn't care.

## What was dropped from the upstream project

| Upstream thing | Why it's gone |
|---|---|
| Flask, SocketIO, routes, templates | Not needed for a local app |
| Flask-Login, Flask-Dance (Google OAuth), JWT worker auth | Single-user desktop context |
| SQLAlchemy, `EggLayingImage` table, custom-mask storage | Images come from the filesystem |
| SMTP error notifications | Local errors print to stderr |
| `project/gpu_backend/worker.py` (separate worker process) | Inference runs in-process |
| `CircleFinder`, `chamber.py`, the `CT` enum, arena model | Replaced by `detect_regions` |
| The arena `.pth` model (`arena_pit_v2.pth`) | See above |
| `unet_reduced_backbone_arena_wells.json` config | Arena model not used |

Everything the pipeline actually computes on is still here: the SplineDist model code, the egg counter config, the outline interpolation logic, and the percentile normalization used before inference.

## What's still needed to count your eggs

- **Your own trained egg counter** (if your eggs look meaningfully different from *Drosophila* eggs). The shipped model was trained on ~2021 lab data. See [../doc/adapting_to_new_images.md](../doc/adapting_to_new_images.md) for notes on retraining — the upstream SplineDist project (https://github.com/uhlmanngroup/splinedist) is where training actually happens; no training code lives in this repo.
- **A working `detect_regions`** for your chamber design. For a smoke test, the default "full image" implementation plus the shipped model will at least run end-to-end.

## API sketch

If you want to call the pipeline from your own Python code (e.g., from a PyQt window):

```python
from pathlib import Path
from eggcount.egg_model import EggModel
from eggcount.pipeline import analyze_image, save_annotated_image, load_image

model = EggModel(weights_path="newapp/models/my_eggs.pth")
result = analyze_image("sample.jpg", model)

print(f"total count: {result.total_count}")
for r in result.regions:
    print(f"  {r.region.label}: {r.count}")

# Draw the annotations and save
img = load_image("sample.jpg")
save_annotated_image(img, result, "sample_annotated.png")

# Or supply your own detector
from eggcount.chamber_detection import Region

def my_detector(img):
    h, w = img.shape[:2]
    # Four fixed square regions, one per corner of the image
    size = min(h, w) // 3
    return [
        Region(bbox=(0, 0, size, size), label="top_left"),
        Region(bbox=(w - size, 0, size, size), label="top_right"),
        Region(bbox=(0, h - size, size, size), label="bottom_left"),
        Region(bbox=(w - size, h - size, size, size), label="bottom_right"),
    ]

result = analyze_image("sample.jpg", model, region_detector=my_detector)
```

## Adding a PyQt GUI

The pipeline is synchronous and takes ~a few seconds per region on GPU. For a GUI, put `analyze_image` on a `QThread` so the event loop stays responsive:

```python
class AnalyzeWorker(QThread):
    finished_ok = pyqtSignal(object)

    def __init__(self, image_path, model):
        super().__init__()
        self.image_path = image_path
        self.model = model

    def run(self):
        result = analyze_image(self.image_path, self.model)
        self.finished_ok.emit(result)
```

Wire `finished_ok` to a slot that updates a `QGraphicsScene` containing the annotated image. That's the whole GUI.
