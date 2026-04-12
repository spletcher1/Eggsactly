# Chamber Types

This document is a deep dive on the `CT` enum and the chamber-type system. If you are considering adding a new chamber type, read [adapting_to_new_images.md](adapting_to_new_images.md) first for the high-level decision tree, then come back here for reference.

## The `CT` enum

`CT` is a Python `enum.Enum` defined at [project/lib/image/chamber.py:212](../project/lib/image/chamber.py#L212). Each enum value is a **class**, not an instance — `CT.opto.value` is `OptoChamber`, and `CT.opto.value()` instantiates it. The rest of the codebase uses `.value()` liberally to grab geometry constants.

```python
class CT(enum.Enum):
    opto        = OptoChamber
    sixByFour   = SixByFourChamber
    fiveByThree = FiveByThreeChamber
    large       = LargeChamber

    # Legacy aliases
    new         = OptoChamber
    old         = SixByFourChamber
    threeBy5    = FiveByThreeChamber
    fourCircle  = LargeChamber
```

The legacy aliases exist to remain compatible with older CSV files and saved data. Identifiers like `CT.opto.name` (the string `"opto"`) are what get stored in `SessionManager.chamberTypes`, serialized in SocketIO events, and compared against in `if` branches.

## The four chamber classes

All four inherit from the `Chamber` base class at [chamber.py:9](../project/lib/image/chamber.py#L9), which provides a default `writeLineFormatted` for CSV output and a static `readCounts` for parsing saved CSVs back in.

| Class | numRows × numCols | rowDist (mm) | colDist (mm) | Agarose constants | Special layout |
|---|---|---|---|---|---|
| [`OptoChamber`](../project/lib/image/chamber.py#L71) | 4 × 5 | 18 | 22 | `dist_across_arena=17`, `dist_between_arenas=5`, `dist_along_agarose=86`, `agarose_width=3.5` | Two agarose strips per well; wells rotated so strips run vertically |
| [`SixByFourChamber`](../project/lib/image/chamber.py#L101) | 6 × 4 | 12 | 25 | `floor_side_length=10`, `dist_between_arenas=3`, `dist_along_agarose=78`, `agarose_width=6` | Two agarose strips per well on opposite sides |
| [`FiveByThreeChamber`](../project/lib/image/chamber.py#L117) | 5 × 3 | 12 | 26 | `floor_side_length=10`, `dist_between_arenas=3`, `dist_along_agarose=64`, `agarose_width=7` | Two agarose strips per well on opposite sides |
| [`LargeChamber`](../project/lib/image/chamber.py#L133) | 2 × 2 | 42 | 42 | `floorSideLength=39` | Four circular agarose wells arranged in a diamond around each of four center points — sixteen crops total |

**`numRepeatedRowsPerCol` and `numRepeatedColsPerRow`** appear on `OptoChamber` and control how the generic `Chamber.writeLineFormatted` reshapes counts into CSV rows. The other Chamber subclasses fall through to defaults of `1` and `2`.

**All the numeric constants are millimeters.** `CircleFinder` measures the detected grid spacing in pixels, divides by `rowDist`/`colDist` to get a pixels-per-mm ratio, and then uses the other constants (multiplied by `pxToMM`) to compute agarose-strip crop sizes. If you add a new chamber class, these constants must reflect the physical geometry of your chamber or the crops will be the wrong size.

### `LargeChamber` is the odd one out

`LargeChamber` bypasses most of the histogram-binning logic in `CircleFinder` and uses OpenCV's `HoughCircles` instead to find the small circular agarose wells around each central point. See [circleFinder.py:239-372](../project/lib/image/circleFinder.py#L239-L372). It also overrides `writeLineFormatted` to produce a diamond-shaped CSV layout and defines two mysterious tuples:

- `dataIndices` ([chamber.py:143](../project/lib/image/chamber.py#L143)) — maps from the sorted-bbox order that `getSubImageBBoxes` produces to the logical "which well is which" order expected by the CSV.
- `csvToClockwise` ([chamber.py:152](../project/lib/image/chamber.py#L152)) — reorders per-well counts so they print in clockwise order (N, E, S, W) in the CSV.

If you are adding a new chamber type and it has more than one agarose region per well, you will need analogous reordering logic.

## `getChamberTypeByRowsAndCols`

Located at [circleFinder.py:132](../project/lib/image/circleFinder.py#L132). Takes the detected `[numRows, numCols]` and iterates over every `CT` member looking for a match. A match can be either exact (`rows == CT.numRows and cols == CT.numCols`) or transposed (`rows == CT.numCols and cols == CT.numRows`). If transposed, it returns `inverted=True`.

```python
for ct in CT:
    if (numRowsCols[0] == ct.value().numRows
        and numRowsCols[1] == ct.value().numCols) \
       or (numRowsCols[0] == ct.value().numCols
           and numRowsCols[1] == ct.value().numRows):
        return (ct.name, <inverted flag>)
return None, False
```

**Important:** because the legacy aliases (`new`, `old`, etc.) all share numRows/numCols values with their canonical counterparts, the first matching `CT` wins. The canonical names are listed first, so a new image with `4×5` wells will always be reported as `"opto"`, not `"new"`.

If you add a new chamber type, make sure your `(numRows, numCols)` tuple is **unique** across the enum, otherwise the inference will assign the wrong label.

## The `inverted` flag

`inverted` means the image's row/column orientation is transposed relative to the "canonical" orientation of the chamber. Physically this usually means the image is rotated 90° compared to how the model expects it. The flag is set by `getChamberTypeByRowsAndCols` and then read in several places to decide which axis to treat as "rows" versus "columns":

| File:Line | What it controls |
|---|---|
| [circleFinder.py:234-236](../project/lib/image/circleFinder.py#L234-L236) | Which `avgDists` index is used as row vs. column spacing when computing `pxToMM` |
| [circleFinder.py:576-601](../project/lib/image/circleFinder.py#L576-L601) | Which of the two `if` branches in `getSubImageBBoxes` runs — i.e., whether agarose strips are considered horizontal or vertical |
| [circleFinder.py:625-626](../project/lib/image/circleFinder.py#L625-L626) | Which index of `numRowsCols` to loop over when producing `sortedBBoxes` |
| [chamber.py:36-41](../project/lib/image/chamber.py#L36-L41) | CSV writer reshape + transpose in `Chamber.writeLineFormatted` |
| [sessionManager.py:306-311](../project/lib/web/sessionManager.py#L306-L311) | Opto label-positioning direction (horizontal vs. vertical offset) |

If you add a chamber type whose rows and columns are visually symmetric, or where inversion doesn't make sense, you can skip supporting `inverted=True` — the flag will simply never be set for that type, and the inverted branches of the above code will never be reached.

## Per-type branches in code

Every place in the codebase that checks `self.ct == CT.<something>.name` or equivalent. If you add a new chamber type, this is the list of places you may need to add a branch for.

### `CircleFinder`

| File:Line | Branch | What it does |
|---|---|---|
| [circleFinder.py:572-573](../project/lib/image/circleFinder.py#L572-L573) | `if self.ct is CT.large.name` | Skips the generic bbox generation and calls `getLargeChamberBBoxesAndImages`, which runs Hough circles to find the four agarose wells per center |
| [circleFinder.py:576-598](../project/lib/image/circleFinder.py#L576-L598) | `if (self.ct is CT.opto.name and not self.inverted) or (self.ct is not CT.opto.name and self.inverted)` | Generates two agarose bboxes per well with the agarose strips oriented **vertically** (`0.5 * avgDists[0]` wide, `8.5 * pxToMM` tall) |
| [circleFinder.py:599-621](../project/lib/image/circleFinder.py#L599-L621) | `elif (self.ct is CT.opto.name and self.inverted) or (self.ct is not CT.opto.name and not self.inverted)` | Generates two agarose bboxes per well with the strips oriented **horizontally** (`8.5 * pxToMM` wide, `0.5 * avgDists[1]` tall) |
| [circleFinder.py:622-623](../project/lib/image/circleFinder.py#L622-L623) | `if self.ct is CT.opto.name` | Delegates to `OptoChamber.getSortedBBoxes()` for the opto-specific bbox reorder |
| [circleFinder.py:627](../project/lib/image/circleFinder.py#L627) | `offset = 4 if self.ct is CT.large.name else 2` | Large produces 4 bboxes per detected center, everyone else produces 2 |
| [circleFinder.py:631-633](../project/lib/image/circleFinder.py#L631-L633) | `if self.ct is CT.large.name` | Append the extra two bboxes per well for large |
| [circleFinder.py:868](../project/lib/image/circleFinder.py#L868) | `if self.ct is not CT.large.name` | Large chamber skips the row/column regression rotation-angle computation |

### `SessionManager`

| File:Line | Branch | What it does |
|---|---|---|
| [sessionManager.py:292-304](../project/lib/web/sessionManager.py#L292-L304) | `if ct == CT.large.name` | Positions labels inside each of the four agarose wells using `i % 4` to pick one of four positional formulas |
| [sessionManager.py:305-311](../project/lib/web/sessionManager.py#L305-L311) | `elif ct == CT.opto.name` | Positions labels with direction-aware offsets depending on the `inverted` flag |
| [sessionManager.py:312-320](../project/lib/web/sessionManager.py#L312-L320) | `else` | Generic case: upper-left corner of the bbox + a small fixed offset, with extra horizontal room if the count is ≥ 10 |
| [sessionManager.py:476-477](../project/lib/web/sessionManager.py#L476-L477) | `CT[self.chamberTypes[imgPath]].value().writeLineFormatted(...)` | Delegates CSV writing to the chamber's own `writeLineFormatted` — no explicit branch here, but each chamber class can override it |

### `chamber.py`

Chamber classes dispatch via method overrides rather than `if` branches. The relevant methods you may need to override in a new chamber class:

| Method | Default in `Chamber` base | Overridden by |
|---|---|---|
| `writeLineFormatted` | Generic row-major reshape; handles `inverted` by transposing | `LargeChamber` (diamond layout) |
| `getSortedBBoxes` | Not present on base class | `OptoChamber` only |
| `flattenCounts` | Not present on base class | `LargeChamber` only |

## Hardcoded magic numbers

This table is the single most important artifact in this doc for anyone adapting to new images. Every numeric constant below ties to the physical geometry of one of the existing chamber types. If the physical geometry of your new chamber differs, these numbers have to be revisited.

| Value | File:Line | Physical meaning | Chamber types affected |
|---|---|---|---|
| `ARENA_IMG_RESIZE_FACTOR = 0.186` | [circleFinder.py:21](../project/lib/image/circleFinder.py#L21) | Downsample factor before feeding the full image to the arena model; tuned to the resolution at which the current arena model was trained | All |
| `bins=40` in `binned_statistic` | [circleFinder.py:651](../project/lib/image/circleFinder.py#L651) | Histogram resolution for finding row/column clusters from scattered detections | All except `large` (which uses Hough circles) |
| `8.5 * pxToMM` | [circleFinder.py:582, 597, 604, 620](../project/lib/image/circleFinder.py#L582) | Half the length of an agarose strip along its long axis (in mm → px) | `opto`, `sixByFour`, `fiveByThree` |
| `4 * pxToMM` | [circleFinder.py:587, 592, 609, 614](../project/lib/image/circleFinder.py#L587) | Offset from well center to the near edge of the agarose strip | `opto`, `sixByFour`, `fiveByThree` |
| `0.5 * avgDists[0]` / `0.5 * avgDists[1]` | [circleFinder.py:581, 586, 591, 596, 605, 610, 615, 620](../project/lib/image/circleFinder.py#L581) | Half the center-to-center spacing between adjacent wells — used as the short-axis half-width of the agarose crop | `opto`, `sixByFour`, `fiveByThree` |
| `minRadius=30, maxRadius=50` | [circleFinder.py:247-248](../project/lib/image/circleFinder.py#L247-L248) | Expected pixel radii for `HoughCircles` detection of small agarose wells | `large` |
| `minDistance=140, param1=40, param2=35` | [circleFinder.py:244-246](../project/lib/image/circleFinder.py#L244-L246) | HoughCircles tuning for the current large-chamber image resolution | `large` |
| `0.5 * 0.25 * floorSideLength * pxToMM` | [circleFinder.py:254](../project/lib/image/circleFinder.py#L254) | Distance threshold for grouping Hough circles to their parent center | `large` |
| `10.5 * pxToMM` | [circleFinder.py:329](../project/lib/image/circleFinder.py#L329) | Diameter of a large-chamber agarose well crop, in mm | `large` |
| `0.5 * 10 * pxToMM` | [circleFinder.py:330](../project/lib/image/circleFinder.py#L330) | Radius of a large-chamber agarose well crop, in mm | `large` |
| `0.1, 0.15, 0.4, 0.2, 0.2, 0.45` (fractions of bbox dimensions) | [sessionManager.py:295-303](../project/lib/web/sessionManager.py#L295-L303) | Label positioning inside large chamber agarose wells — one formula per of the four positions | `large` |
| `1.4, -0.1` (fractions of bbox dimensions) | [sessionManager.py:307-310](../project/lib/web/sessionManager.py#L307-L310) | Label positioning offset for opto — labels sit outside the bbox on alternating sides | `opto` |
| `50, textLabelHeight=96` | [sessionManager.py:319-320, 68](../project/lib/web/sessionManager.py#L319-L320) | Generic label offset (pixels) from the upper-left corner of each bbox | `sixByFour`, `fiveByThree`, anything that falls through to the default branch |

### Reading the bbox arithmetic

The `getSubImageBBoxes` bboxes are stored as `[x, y, width, height]`. The idiom used throughout is:

```python
bboxes.append([
    max(center[0] - int(<x_offset>), 0),   # x = clipped upper-left corner
    max(center[1] - int(<y_offset>), 0),   # y = clipped upper-left corner
])
bboxes[-1] += [
    center[0] + int(<x_extent>) - bboxes[-1][0],   # width  = absolute right edge - x
    center[1] + int(<y_extent>) - bboxes[-1][1],   # height = absolute bottom edge - y
]
```

The upper-left corner is clamped to `(0, 0)` with `max(..., 0)`, but the width/height are computed from the *unclamped* target right/bottom edges — so a bbox that would extend off the top-left of the image ends up with the correct right/bottom edge but a smaller-than-expected width/height. If you adapt this code, be aware of that asymmetry.

## Where custom-mask geometry lives

If a user draws a custom chamber mask in the browser UI, that path bypasses `CircleFinder` entirely and goes through [project/lib/image/node_based_segmenter.py](../project/lib/image/node_based_segmenter.py). That module has its own per-`CT` branches for px-to-mm measurement and bbox generation — see the `calc_px_to_mm_ratio` and `divide_img*` methods. It is an escape hatch for one-off analyses but is not a scalable adaptation path. See [adapting_to_new_images.md](adapting_to_new_images.md#escape-hatch-custom-masks) for when it's the right tool.
