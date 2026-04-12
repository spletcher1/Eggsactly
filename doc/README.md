# Eggsactly Developer Documentation

This directory contains **developer-facing documentation**: how the system is built internally and how to adapt it to new image structures. For installation, usage, and demo instructions, see the top-level [README.md](../README.md).

## Guides

- **[adapting_to_new_images.md](adapting_to_new_images.md)** — Step-by-step adaptation guide. **Start here if you're trying to point Eggsactly at images with a different structure than the *Drosophila* egg-laying chambers it was built for.** Includes a decision tree, three adaptation paths (add a chamber type / retrain arena detector / retrain egg counter), and a worked example.
- **[architecture.md](architecture.md)** — End-to-end pipeline map: how an uploaded image becomes an egg count. Covers the HTTP/SocketIO surface, the GPU task queue, the worker process, `CircleFinder` post-processing, `SessionManager` orchestration, and frontend coupling.
- **[chamber_types.md](chamber_types.md)** — The `CT` enum system in depth. Every chamber class, every per-type branch in `CircleFinder` and `SessionManager`, every hardcoded magic number tied to chamber geometry. Reference material for anyone editing the chamber-type layer.
- **[models_and_configs.md](models_and_configs.md)** — The two ML models, their config JSONs, the `SplineDist2D` architecture, and what retraining requires. Flags that **no training code lives in this repository** — the `.pth` files are produced externally.

## Quick pointers

| If you want to... | Read this |
|---|---|
| Understand the overall request flow | [architecture.md](architecture.md) |
| Add support for a new chamber layout | [adapting_to_new_images.md](adapting_to_new_images.md) → Path A |
| Swap in a retrained arena or egg model | [adapting_to_new_images.md](adapting_to_new_images.md) → Path B or C, then [models_and_configs.md](models_and_configs.md) |
| Find every place a specific chamber type is special-cased | [chamber_types.md](chamber_types.md#per-type-branches-in-code) |
| Debug a `ValueError: zero-size array` from scipy | [adapting_to_new_images.md](adapting_to_new_images.md#where-to-look-when-things-go-wrong) |
| Understand the `inverted` flag | [chamber_types.md](chamber_types.md#the-inverted-flag) |
