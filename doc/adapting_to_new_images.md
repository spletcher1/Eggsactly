# Adapting Eggsactly to New Images

**Start here if you're trying to point Eggsactly at images with a different structure than the *Drosophila* egg-laying chambers it was built for.**

This is the practical adaptation guide. It's opinionated about where to put your effort. The companion docs ([architecture.md](architecture.md), [chamber_types.md](chamber_types.md), and [models_and_configs.md](models_and_configs.md)) are reference material — come back to them when you need exact file:line anchors.

## The honest answer

You asked: "Is this as easy as defining a new chamber type?"

**In the general case, no.** Defining a new `CT` enum value is necessary, but rarely sufficient. Four separate layers have assumptions baked in about the existing chamber geometries, and at least the first of the four is almost always a blocker:

1. **The arena-detection model** (`arena_pit_v2.pth`) only recognizes things that look like the wells it was trained on. If your wells are shaped differently, sized differently, photographed at a different zoom, or sit on a visually different background, the model will return zero detections and `CircleFinder` will crash downstream. This is the bug you are probably hitting right now if you're reading this.
2. **`CircleFinder.getSubImageBBoxes`** has hardcoded `8.5 * pxToMM` and `4 * pxToMM` offsets that encode the physical size of the agarose strips next to each well. These numbers are correct for the existing chamber types and wrong for anything else.
3. **`CircleFinder.processDetections`** assumes wells form a regular grid recoverable by histogram binning along X and Y. Hexagonal, radial, or irregular layouts will fail.
4. **`SessionManager.send_annotations_for_task`** has per-`CT` label-positioning branches — minor, but they exist.

The good news: **the frontend is data-driven.** Once the server emits the right bounding boxes and counts, the browser will draw them correctly with zero JavaScript changes.

## Decision tree

Answer these questions in order. Stop at the first one that matches.

```
Q1. Are your images visually similar to the existing chambers — circular wells
    arranged in a regular rectangular grid, roughly the same scale?
    │
    ├── YES ──▶ Path A: Add a new chamber type. No retraining.
    │            Expect to edit: chamber.py, circleFinder.py,
    │            possibly sessionManager.py.
    │
    └── NO ───▶ Q2. Are you still counting things that look like fly eggs?
                 │
                 ├── YES ──▶ Path B: Retrain the arena detector.
                 │            Keep the egg counter as-is.
                 │
                 └── NO ───▶ Path C: Retrain both models.
```

**Short-circuit option:** if you just need to process a small number of images one-off and don't want to write code at all, see [the escape hatch](#escape-hatch-custom-masks) below — you can draw chamber masks by hand in the browser UI.

## Path A — Add a new chamber type

Use this path if your images have a **new grid layout but similar physical wells**. Example: same well size and shape as the existing opto chamber, but 3×3 instead of 4×5.

### Step 1. Define the chamber class

Add a new subclass of `Chamber` in [project/lib/image/chamber.py](../project/lib/image/chamber.py) alongside the existing four. Use `SixByFourChamber` at [chamber.py:101](../project/lib/image/chamber.py#L101) as a template for the common case (two agarose strips per well).

```python
class ThreeByThreeChamber(Chamber):
    """Represent a 3x3 grid chamber type."""

    def __init__(self):
        self.numRows, self.numCols = 3, 3
        self.rowDist, self.colDist = 15, 20  # millimeters, adjust to your layout
        self.floor_side_length = 10
        self.dist_between_arenas = 3
        self.dist_along_agarose = 78
        self.agarose_width = 6
        self.dist_between_floors = 2 * self.agarose_width + self.dist_between_arenas
        self.dist_trough_to_first_arena = 4
```

**Every constant is in millimeters.** Measure them off a physical chamber or an accurately scaled image. Getting them wrong won't crash anything — it will silently produce mis-cropped sub-images, which wastes model effort and gives bad counts.

If your chamber has a different CSV-output layout than a plain row-major grid, override `writeLineFormatted`. Use `LargeChamber.writeLineFormatted` at [chamber.py:162](../project/lib/image/chamber.py#L162) as a template for custom layouts. For a plain grid, the default in the base class is fine.

### Step 2. Register it in the `CT` enum

Add an entry to the enum at [chamber.py:212](../project/lib/image/chamber.py#L212):

```python
class CT(enum.Enum):
    opto         = OptoChamber
    sixByFour    = SixByFourChamber
    fiveByThree  = FiveByThreeChamber
    large        = LargeChamber
    threeByThree = ThreeByThreeChamber   # ← new

    # Legacy aliases
    new          = OptoChamber
    old          = SixByFourChamber
    threeBy5     = FiveByThreeChamber
    fourCircle   = LargeChamber
```

**Check the `(numRows, numCols)` uniqueness.** `getChamberTypeByRowsAndCols` returns the first matching enum member, so your new `(3, 3)` must not clash with any existing type's rows/cols (it doesn't, but always verify). See [chamber_types.md](chamber_types.md#getchambertypebyrowsandcols) for details.

### Step 3. Add a bbox branch in `CircleFinder.getSubImageBBoxes`

Open [project/lib/image/circleFinder.py:554](../project/lib/image/circleFinder.py#L554) and find the existing per-type `if`/`elif` chain around lines 572–621.

If your new chamber type **uses the same two-agarose-strips-per-well geometry** as `sixByFour` / `fiveByThree`, the existing `elif` branch already handles it — but only because of this condition:

```python
elif (self.ct is CT.opto.name and self.inverted) or (
      self.ct is not CT.opto.name and not self.inverted
):
```

Your type is "not opto," so the second clause matches and you fall through. **Verify that the `8.5 * pxToMM` and `4 * pxToMM` offsets are physically correct for your agarose-strip dimensions.** If they aren't, copy the branch, gate it on `self.ct is CT.threeByThree.name`, and substitute your own offsets computed from your chamber's constants.

Example pattern for a custom branch:

```python
elif self.ct is CT.threeByThree.name:
    half_across = int(0.5 * my_dist_across_arena * pxToMM)
    half_along  = int(0.5 * self.chamber.dist_along_agarose * pxToMM)
    edge_offset = int(0.5 * self.chamber.dist_between_arenas * pxToMM)
    for center in centers:
        # left agarose strip
        bboxes.append([...])
        # right agarose strip
        bboxes.append([...])
```

Also check [circleFinder.py:627](../project/lib/image/circleFinder.py#L627) — the `offset = 4 if self.ct is CT.large.name else 2` line. If your type produces more or fewer than 2 bboxes per well, this needs to change.

### Step 4. Label positioning (usually optional)

Open [project/lib/web/sessionManager.py:292](../project/lib/web/sessionManager.py#L292). The `if ct == CT.large.name / elif ct == CT.opto.name / else` chain decides where to put each count label relative to its bbox. **The generic `else` branch** ([line 312](../project/lib/web/sessionManager.py#L312)) writes the label in the upper-left corner of the bbox and shifts right if the count is ≥ 10. For most new chamber types this is fine — skip this step and the default will work.

If you want custom label positioning, add an `elif ct == CT.threeByThree.name:` branch computing the `(x, y)` you want.

### Step 5. Try it on a real image

Start the dev server, upload a sample image, and watch:

- **Server stdout** for `exception while finding circles:` — if you see this, the arena model didn't find your wells. Go to Path B.
- **Browser DevTools** → SocketIO frames → look for `chamber-analysis` and verify `chamberType` is `"threeByThree"` and `bboxes` has the expected length and positions.
- **Browser UI** — the overlay should draw bboxes in the right places visually. If bboxes are offset or the wrong size, your constants in Step 1 or offsets in Step 3 are off.

If all of the above looks right but egg counts are nonsense, your sub-image crops are off: revisit Step 3 with corrected offsets. If the crops look fine but counts are still off, the egg counter is disagreeing with you about what an egg looks like — either your background is throwing it off or Path C is needed.

### When Path A is not enough

You will know you need to move to Path B (retraining) if **after Step 3 you still see zero detections or consistently wrong well centers.** That means the arena model isn't producing useful outputs for your images — which code changes alone cannot fix.

## Path B — Retrain the arena detector

Use this path if the existing arena model cannot find your wells (Path A step 5 fails), or if your well **shape** (not just layout) differs from the existing circular wells.

### What you need

1. **A labeled training set.** You need images of your chambers with every well outlined. SplineDist expects label images where each well is a unique non-zero integer ID on a background of zero. For a usable model you'll want at least **a few hundred labeled wells** spanning the lighting and angle variations your users will encounter.
2. **The upstream SplineDist training code** from https://github.com/uhlmanngroup/splinedist. This repo (Eggsactly) is inference-only — see [models_and_configs.md](models_and_configs.md#no-training-code-lives-in-this-repo).
3. **A GPU** for training (CUDA-capable; same requirement as inference).

### Retraining steps

1. **Prepare your dataset** in the SplineDist format (outline-labeled images).
2. **Clone and set up upstream SplineDist.** Follow their README for environment setup.
3. **Create a training config.** Start by copying Eggsactly's [unet_reduced_backbone_arena_wells.json](../project/configs/unet_reduced_backbone_arena_wells.json) and tweaking the relevant keys:
   - `backbone`: `"unet_reduced"` is ~3x faster than `"unet_full"`; stick with reduced unless accuracy is insufficient.
   - `train_patch_size`: must be ≥ the size of your largest labeled well in pixels. Bigger means more GPU memory.
   - `n_control_points`: 8 is fine for roughly circular shapes. Increase to 12–16 if your wells are highly non-round.
   - `train_epochs`: 400 works for the existing training set. Start there; reduce if you're overfitting.
4. **Train** using the upstream SplineDist training script. Target output format must match what `SplineDist2D.predict_instances` returns: a dict with `points`, `prob`, and `coord` keys. If you used the upstream trainer, this is automatic.
5. **Export** the final `.pth` file.

### Dropping the new model into Eggsactly

1. **Copy the `.pth`** to `project/models/arena_<your_name>.pth`.
2. **Update `NETWORK_CONSTS`** in [project/gpu_backend/worker.py](../project/gpu_backend/worker.py) to point at the new file and the new config JSON (if different from the existing one).
3. **Also update the in-process arena model load** in [circleFinder.py:22-37](../project/lib/image/circleFinder.py#L22-L37) — both places load the arena model and both need the same `.pth` path.
4. **Tune `ARENA_IMG_RESIZE_FACTOR`** at [circleFinder.py:21](../project/lib/image/circleFinder.py#L21) to whatever downsample factor your training images used. This is a hardcoded scale — see [models_and_configs.md](models_and_configs.md#the-arena_img_resize_factor-constant).

### If your wells are not circular

The existing pipeline assumes circular wells at two specific points:

- [circleFinder.py:54](../project/lib/image/circleFinder.py#L54): `fit_circle_kasa` does a least-squares circle fit to each predicted outline and uses the fitted center as the well centroid. For non-circular wells, replace this with either `np.mean(outline, axis=0)` (centroid of the outline points) or — simpler — skip the outline step entirely and use `predictions["points"]` directly. The SplineDist `points` are already predicted centroids, so they should work even if `coord` describes a non-circular shape.
- [circleFinder.py:239-372](../project/lib/image/circleFinder.py#L239-L372): `findAgaroseWells` / `getLargeChamberBBoxesAndImages` use `cv2.HoughCircles`, which is fundamentally a circle detector. This only runs for `CT.large`, so you only need to worry about it if your new chamber mimics the large-chamber geometry (multiple sub-wells per center).

### Still do Path A too

Retraining doesn't replace the work in Path A — you still need to define the chamber class, register the CT entry, and make sure `getSubImageBBoxes` produces the right crops. Path B just replaces the assumption "arena_pit_v2.pth already knows how to find wells" with "my retrained model knows how to find wells."

## Path C — Retrain the egg counter

Use this path if you're counting something that isn't a fly egg — say, a different organism's eggs, or cell nuclei, or something entirely unrelated.

Steps are the same as Path B but for the egg model:

1. **Labeled training set** of cropped sub-images with each countable object outlined.
2. **Training config**: start from [unet_backbone_rand_zoom.json](../project/configs/unet_backbone_rand_zoom.json). Keep `backbone: "unet_full"` unless you have a strong speed reason to go reduced. Set `train_patch_size` to match the typical size of your sub-image crops (so training sees full-context crops, not tiny patches).
3. **Train** using upstream SplineDist.
4. **Drop the new `.pth`** into `project/models/` and update `NETWORK_CONSTS` in [worker.py](../project/gpu_backend/worker.py). Unlike the arena model, the egg model is only loaded in the worker — no second load site.
5. **Update `modelRevDates.json`** ([project/models/modelRevDates.json](../project/models/modelRevDates.json)) with the new filename and a new date. This affects what CSV exports report as the model version.

Note that the egg model does **not** have its own `ARENA_IMG_RESIZE_FACTOR` equivalent — egg sub-images are fed to the model at native resolution. So the training crop size and your expected sub-image size should be aligned.

## Escape hatch: custom masks

If you're adapting as a one-off for a single experiment and don't want to touch code at all, there's a separate code path: users can draw their own chamber mask in the browser UI and save it. This goes through [project/lib/image/node_based_segmenter.py](../project/lib/image/node_based_segmenter.py) instead of `CircleFinder`.

**When to use it:**
- You have a handful of images to process.
- Each one can have its mask drawn by hand without being a miserable experience.
- You're willing to rely on the existing egg-counting model (i.e., you're still counting fly eggs in images with similar lighting to the training data).

**When not to use it:**
- You need to process many images automatically.
- You want reproducibility across users.
- The new image type will be used long-term.

The custom mask system has its own per-`CT` branches for px-to-mm measurement and bbox generation, so it's not fully chamber-agnostic — but it is more flexible for irregular layouts than the histogram-binning approach in `CircleFinder`.

## Validation checklist

After any adaptation, walk through this:

- [ ] **Server starts cleanly.** No `FileNotFoundError` for missing weights, no import errors from new code.
- [ ] **Worker starts cleanly.** The GPU worker process (separate from Flask) loads both models without CUDA memory errors.
- [ ] **Upload a sample image.** Watch server stdout for `exception while finding circles:` — if you see it, the arena model isn't finding your wells.
- [ ] **`chamber-analysis` SocketIO frame.** In browser DevTools → Network → WS, find the `chamber-analysis` message and confirm `chamberType` is what you expect and `bboxes.length` matches the number of wells in your chamber.
- [ ] **Visual overlay.** The chamber outlines drawn in the browser match the physical wells in the image. Off-center or wrong-size bboxes usually mean the mm constants in your Chamber subclass are off.
- [ ] **Spot-check egg counts.** Manually count eggs in 3–5 regions and compare to what the model reports. Large discrepancies mean Path C is needed — or that your sub-image crops are not lined up correctly.
- [ ] **CSV export.** Request a CSV and verify the layout is reasonable. If you added a custom `writeLineFormatted`, this is where you'll see whether the reshape logic works.

## Where to look when things go wrong

- **TypeError or missing-kwarg errors** inside task completion: see the existing `**kwargs` in [sessionManager.py:134](../project/lib/web/sessionManager.py#L134) — listeners receive all keys from the worker's result dict, so your new handlers should either accept all or use `**kwargs`.
- **`ValueError: zero-size array to reduction operation minimum`** from scipy: this is the "arena model returned no detections" failure mode. The exception is caught at [sessionManager.py:183-201](../project/lib/web/sessionManager.py#L183-L201) and reported as an `ImageAnalysisException` — the server stays up, but the image is unusable. This almost always means you need Path B.
- **`ConnectionRefusedError` from SMTP**: unrelated to chamber adaptation — the error-notification email handler tries to contact a local SMTP server. The user-configured `send_mail` now skips silently when no recipients are configured and swallows SMTP errors; this should no longer crash threads.
- **Bboxes drawn in the right places but with the wrong size**: your `pxToMM` is computed correctly but the offsets in your `getSubImageBBoxes` branch use the wrong constants. Double-check Step 3 of Path A.
- **Bboxes drawn in the wrong places entirely**: `getChamberTypeByRowsAndCols` is returning the wrong chamber type, or the `inverted` flag is wrong. Add a print statement to `processDetections` to see what `numRowsCols` is detecting.
