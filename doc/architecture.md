# Architecture Overview

This document explains how an uploaded image becomes an egg count. It is intended for developers who need to modify Eggsactly, not end users. For installation and usage, see the top-level [README.md](../README.md).

## Pipeline at a glance

```
┌──────────┐   HTTP upload   ┌──────────────┐   enqueue    ┌─────────────┐
│ Browser  │ ──────────────▶ │ Flask app    │ ───────────▶ │ GPUManager  │
│ (counting│                 │ (main.py,    │              │ task queue  │
│  .html)  │ ◀─SocketIO────  │  socket_     │              └──────┬──────┘
└──────────┘     events      │  events.py)  │                     │
                             └──────┬───────┘                     │ poll
                                    │                             ▼
                                    │                      ┌─────────────┐
                                    │         POST results │ GPU worker  │
                                    │     ◀──────────────── │ (separate   │
                                    │                       │  process)   │
                                    ▼                       └──────┬──────┘
                           ┌──────────────────┐                    │
                           │ SessionManager   │                    │
                           │  (per-room       │                    │
                           │   orchestration) │                    │
                           └──────────────────┘                    │
                                                                   │
                                                                   ▼
                                                           ┌─────────────┐
                                                           │ Arena model │
                                                           │ (first)     │
                                                           └──────┬──────┘
                                                                  │
                                                                  ▼
                                                           ┌─────────────┐
                                                           │ CircleFinder│
                                                           │ post-       │
                                                           │ processing  │
                                                           └──────┬──────┘
                                                                  │
                                                                  ▼
                                                           ┌─────────────┐
                                                           │ Egg model   │
                                                           │ (per crop)  │
                                                           └─────────────┘
```

The pipeline has two ML stages. The first (**arena model**) runs once per uploaded image and finds the locations of the wells. The second (**egg model**) runs once per well sub-image and counts eggs inside it. Post-processing between the two stages is handled by `CircleFinder`, which converts raw model outputs into structured row/column grids and crops.

## HTTP / SocketIO surface

**Upload endpoint.** File uploads are handled in [project/routes/main.py](../project/routes/main.py). Uploaded images are written either to the filesystem under `uploads/<room>/` or to the SQL database as `EggLayingImage` blobs, depending on the configured backend.

**SocketIO events.** Real-time progress and results flow over SocketIO. Registration happens in [project/routes/socket_events.py](../project/routes/socket_events.py). Key events:

| Direction | Event | Purpose |
|---|---|---|
| Client → Server | `connect` | Creates a `SessionManager` for the new room |
| Client → Server | `prepare-counts-csv` | Requests the CSV export of current counts |
| Client → Server | `save-custom-mask` / `load-custom-mask` | Custom chamber masks (escape hatch — see [adapting_to_new_images.md](adapting_to_new_images.md#escape-hatch-custom-masks)) |
| Client → Server | `email-error-notification` | Notifies server of a client-side analysis error |
| Server → Client | `counting-progress` | Free-text progress updates during analysis |
| Server → Client | `chamber-analysis` | Detected chamber type + bounding boxes for one image |
| Server → Client | `counting-annotations` | Per-region egg counts + label positions |
| Server → Client | `counting-error` | Analysis failed for a specific image |
| Server → Client | `counting-csv` | CSV payload for download |

The client (see [project/templates/counting.html](../project/templates/counting.html)) subscribes to these events and draws whatever the server sends — it has no per-chamber-type rendering logic of its own.

## GPU task queue

The GPU workers run in a **separate process** from the Flask app, poll the app for work, and POST results back. The queue abstraction lives in three small modules:

- [project/lib/web/gpu_task_types.py](../project/lib/web/gpu_task_types.py) defines `GPUTaskTypes.arena` and `GPUTaskTypes.egg` — only two task types exist.
- [project/lib/web/gpu_task_group.py](../project/lib/web/gpu_task_group.py) is a group of related tasks (e.g., one egg-counting task group contains one task per uploaded image). When all tasks in a group complete, the group fires an `on_completion` event carrying aggregated results.
- [project/lib/web/gpu_manager.py](../project/lib/web/gpu_manager.py) owns the queue and the map of active task groups.

Workers poll [project/routes/tasks.py:78](../project/routes/tasks.py#L78) (`GET /tasks/gpu`) to pick up work and POST results to `/tasks/gpu/<group_id>` at [project/routes/tasks.py:133](../project/routes/tasks.py#L133). The result handler spawns a `TaskFinalizer` thread which calls `register_completed_task`, which in turn fires the task group's completion event. That event's listener — set up earlier by `SessionManager` — is what actually kicks off the next stage of the pipeline.

## GPU worker

[project/gpu_backend/worker.py](../project/gpu_backend/worker.py) is the inference process. At startup it:

1. Loads both `.pth` files into CUDA memory via `init_splinedist_network` (see the `NETWORK_CONSTS` dict near the top of the file).
2. Authenticates to the Flask app using a JWT signed with a private key it holds locally.
3. Enters a polling loop, picking up one task at a time.

For an **arena task**, the worker:
1. Fetches the full-resolution image.
2. Resizes it by `ARENA_IMG_RESIZE_FACTOR = 0.186` (defined in [project/lib/image/circleFinder.py:21](../project/lib/image/circleFinder.py#L21)). This factor is tuned to the resolution at which the current arena model was trained — if you retrain on different-resolution images you will need to change it.
3. Runs `networks[GPUTaskTypes.arena].predict_instances(...)`, which returns a dict with `points`, `prob`, and `coord` (spline control-point) keys.
4. Returns `{"predictions": [...], "metadata": {...}}` to the Flask app.

For an **egg task**, the worker:
1. Uses `SubImageHelper` to crop out the single sub-image it was told to process.
2. Runs the egg model on the crop.
3. Returns prediction outlines and a count.

## CircleFinder: arena post-processing

[project/lib/image/circleFinder.py](../project/lib/image/circleFinder.py) takes the raw arena-model output and turns it into a clean grid of bounding boxes. This is where most of the per-chamber-type geometry lives.

The happy-path flow through `CircleFinder.findCircles()`:

1. **Fit circles to raw outlines.** For each predicted spline outline, `fit_circle_kasa` ([line 54](../project/lib/image/circleFinder.py#L54)) does a least-squares algebraic circle fit and yields `(cx, cy, r)`. This assumes the wells are circular — a load-bearing assumption when adapting to non-circular well shapes.
2. **Organize centroids into rows and columns.** `processDetections()` ([line 636](../project/lib/image/circleFinder.py#L636)) histograms the X and Y coordinates of all detected centroids into 40 bins using `scipy.stats.binned_statistic`, then finds "true regions" (contiguous non-empty bins). Each cluster of non-empty bins becomes one row (for the Y histogram) or one column (for the X histogram). Missing detections are interpolated by linear regression across the grid.
3. **Infer chamber type.** Once it knows how many rows and columns were detected, it calls `getChamberTypeByRowsAndCols()` ([line 132](../project/lib/image/circleFinder.py#L132)) to match `(rows, cols)` against the known `CT` enum values. If rows/cols are swapped relative to the expected layout, the `inverted` flag is set.
4. **Generate sub-image bounding boxes.** `getSubImageBBoxes()` ([line 554](../project/lib/image/circleFinder.py#L554)) produces the bounding box for each agarose strip or well. This is where the hardcoded `8.5 * pxToMM` / `4 * pxToMM` / `0.5 * avgDists[...]` offsets live. See [chamber_types.md](chamber_types.md) for the full table.

### Why the binning approach fails on empty detections

This is the exact bug the project just hit. If the arena model returns zero outlines — because it wasn't trained on your image style, or the image isn't similar enough to the training set — then `self.centroids` is empty, `self.detections` is two empty arrays, and `binned_statistic` blows up with `ValueError: zero-size array to reduction operation minimum which has no identity` when it tries to call `.min()` on an empty array.

The exception is caught by [project/lib/web/sessionManager.py:183-201](../project/lib/web/sessionManager.py#L183-L201) and reported to the client as an `ImageAnalysisException`, so the server doesn't die — but the image analysis fails. If you're seeing this error it almost always means **the arena model didn't find any wells in your image**, which usually means your images are sufficiently different from the training data that the existing model is unusable.

## SessionManager: per-room orchestration

[project/lib/web/sessionManager.py](../project/lib/web/sessionManager.py) owns a per-socket-room state object and orchestrates the two-stage inference for a given upload batch.

The flow it implements:

1. **`check_chamber_type_and_find_bounding_boxes()`** is called once per uploaded image. It reads the image shape, constructs a `CircleFinder`, and enqueues an arena task group.
2. When the arena task completes, **`segment_image_via_object_detection()`** ([line 134](../project/lib/web/sessionManager.py#L134)) runs `findCircles` on the returned model outputs, stores the detected chamber type and bboxes, and emits `chamber-analysis` to the room.
3. The client acknowledges by sending one `segment-img-and-count-eggs` call per image. **`segment_img_and_count_eggs()`** ([line 228](../project/lib/web/sessionManager.py#L228)) enqueues egg-counting tasks into a single task group, then sets up a completion listener.
4. When the whole egg task group completes, **`send_annotations_for_task_group()`** ([line 339](../project/lib/web/sessionManager.py#L339)) iterates over the per-image predictions and calls **`send_annotations_for_task()`** ([line 259](../project/lib/web/sessionManager.py#L259)). This is where per-chamber-type label positioning lives — see [chamber_types.md](chamber_types.md#per-type-branches-in-code).

Errors are caught and funneled through `report_counting_error()` ([line 97](../project/lib/web/sessionManager.py#L97)), which emits `counting-error` to the room. When an error is a `CUDAMemoryException` (OOM during inference), it's re-raised so the worker can retry with reduced batch size.

## Frontend coupling

[project/templates/counting.html](../project/templates/counting.html) is the single-page counting UI. It uses paper.js for the overlay layer and receives data via SocketIO.

**It is data-driven.** When the server emits `chamber-analysis` the client stores the chamber-type string but only uses it as a bookkeeping label — the actual rendering just draws whatever bboxes the server sent. When `counting-annotations` arrives, the client parses the JSON payload and draws text labels at whatever `(x, y)` positions the server computed. Label positioning is entirely server-side in `send_annotations_for_task()`.

This matters for adaptation: **if you add a new chamber type, you will almost certainly not need to touch any JavaScript or HTML.** The coupling between frontend and chamber geometry is through the wire format only.

## Summary map

| Concern | File | Why it matters for adaptation |
|---|---|---|
| HTTP uploads | [project/routes/main.py](../project/routes/main.py) | Unlikely to change |
| SocketIO events | [project/routes/socket_events.py](../project/routes/socket_events.py) | Unlikely to change |
| Task queue | [project/lib/web/gpu_manager.py](../project/lib/web/gpu_manager.py), [gpu_task_group.py](../project/lib/web/gpu_task_group.py), [gpu_task_types.py](../project/lib/web/gpu_task_types.py) | Unlikely to change |
| Worker / inference | [project/gpu_backend/worker.py](../project/gpu_backend/worker.py) | Change `NETWORK_CONSTS` if swapping model weights |
| Arena post-processing | [project/lib/image/circleFinder.py](../project/lib/image/circleFinder.py) | **High-touch.** Per-chamber-type branches live here |
| Chamber type definitions | [project/lib/image/chamber.py](../project/lib/image/chamber.py) | **The natural extension point.** Add a new class + CT entry |
| Orchestration + labels | [project/lib/web/sessionManager.py](../project/lib/web/sessionManager.py) | Per-chamber-type label positioning at [lines 292-320](../project/lib/web/sessionManager.py#L292-L320) |
| Custom masks | [project/lib/image/node_based_segmenter.py](../project/lib/image/node_based_segmenter.py) | Non-code adaptation path for prototypes |
| Frontend | [project/templates/counting.html](../project/templates/counting.html) | Data-driven; unlikely to change |
