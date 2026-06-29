# ComfyUI Graph-Walk — IIB parser plugin

A drop-in parser plugin for
[sd-webui-infinite-image-browsing](https://github.com/zanllp/sd-webui-infinite-image-browsing)
(IIB) that correctly reads generation metadata from **ComfyUI** images.

## Why

IIB's built-in ComfyUI parser only handles simple, linear workflows and fails silently
on anything more complex, producing wrong or missing model name, prompts, seed, steps,
CFG scale and image size.

This plugin replaces it. It modifies no core IIB files (uninstall by deleting this
folder) and never writes to your image files — all metadata is read live from the
workflow graph embedded in each image at scan time.

## Requirements

- **Python ≥ 3.10** (uses PEP 604 `X | None` type hints)
- A reasonably recent **IIB** with the plugin loader (`plugins/` directory). The plugin
  imports a few IIB internals, including one private helper for a non-standard EXIF
  fallback; that helper's absence degrades only that fallback, not the whole plugin.
  Tested against [IIB v1.8.0](https://github.com/zanllp/infinite-image-browsing/releases/tag/v1.8.0).

## Install

1. Copy this `comfyui_graphwalk/` folder into IIB's `plugins/` directory.
2. **Restart IIB** (plugins load only at startup). The console prints
   `IIB loaded plugin: ComfyUI Graph-Walk`.

Every **newly scanned** image is now parsed correctly.

## Refresh already-indexed images (one-time)

IIB caches parsed metadata in its SQLite DB (`iib.db`), so already-indexed images keep
their old (wrong) info until the cache is invalidated. Run **one** of the following.

### Recommended — targeted invalidation (preserves the rest of the cache)

Stop IIB, then run (uses IIB's bundled Python; adjust the `iib.db` path if needed):

```bash
python -c "import sqlite3,os; db=os.path.expanduser('~/sd-webui-infinite-image-browsing/iib.db'); c=sqlite3.connect(db); c.execute(\"DELETE FROM image_tag WHERE image_id IN (SELECT id FROM image WHERE exif LIKE '%Source Identifier: ComfyUI%' AND exif_edited=0)\"); n=c.execute(\"DELETE FROM image WHERE exif LIKE '%Source Identifier: ComfyUI%' AND exif_edited=0\").rowcount; c.commit(); print('invalidated:', n)"
```

Then start IIB and browse the folder, or click **"Rebuild image index"** once, to
re-index the removed rows. Only ComfyUI entries are touched, manually edited entries
(`exif_edited=1`) and their tags are preserved, and the command is safe to re-run.

### Simplest — full reset

Stop IIB, delete the DB, restart. Re-scans everything; **custom tags are lost**.

```bash
rm ~/sd-webui-infinite-image-browsing/iib.db
```

### Not recommended

- **`touch`-ing image files** forces a re-scan but destroys original mtimes (breaks
  date sorting).
- **"Rebuild image index" alone** rebuilds tags but does **not** refresh the cached
  `exif`, so detail-panel info stays stale. Pair it with the invalidation above.

## How it works

Each ComfyUI image embeds its full workflow graph. The plugin walks that graph
backwards from the sampler node — through LoRA loaders, Reroutes, conditioning merges
and other intermediates — until it reaches the source values (checkpoint name, CLIP
encoders, latent size, seed/steps/CFG scalars). These are assembled into an A1111-style
`parameters` string and passed to IIB's `parse_generation_parameters`, so tags, search
and "Send to txt2img" keep working.

Prepended to IIB's parser list, it transparently replaces the built-in ComfyUI parser;
other image types (A1111, Fooocus, …) are untouched. PNG, WEBP and JPEG are supported.

Coverage is broad but not exhaustive — workflows using node types not listed below may
produce incomplete results.

## Supported nodes

### Sampler entry points

| Node | Pack |
|---|---|
| `KSampler` | built-in |
| `KSamplerAdvanced` | built-in |
| `KSampler (Efficient)` | efficiency-nodes |
| `KSamplerAdvanced (Efficient)` | efficiency-nodes |
| `Tiled KSampler` | ComfyUI-TiledDiffusion |
| `BNK_Unsampler` | ComfyUI-BNK |

### Prompt text — CLIP encoders

Reached by following the `positive` / `negative` conditioning edges.

| Node | Notes |
|---|---|
| `CLIPTextEncode` | standard SD 1.x / 2.x |
| `CLIPTextEncodeSDXL` | reads both `text_g` (global) and `text_l` (local) |
| `CLIPTextEncodeSDXLRefiner` | |
| `smZ CLIPTextEncode` | smZNodes |
| `BNK_CLIPTextEncodeAdvanced` | ComfyUI-BNK |

### Prompt text — text resolution chain

Followed recursively from a text input until a string is produced.

| Node | Behaviour |
|---|---|
| `Reroute` | transparent wire redirect |
| `ImpactWildcardProcessor` | reads `populated_text` (expanded wildcards at `mode='reproduce'`) |
| `RegexReplace` | follows upstream text, then applies the regex substitution |
| `StringFunction` / `StringFunction\|pysssss` | concatenates `text_a`, `text_b`, `text_c` |
| Primitive / `ShowText` / `Note` / … | any node storing a scalar in `value`, `text`, or `string` |

### Conditioning

- **Passthrough** (continue upstream): `Reroute` · `ConditioningSetTimestepRange` ·
  `ConditioningSetArea` · `ConditioningSetAreaPercentage` · `ConditioningSetMask`
- **Merge** (both inputs joined): `ConditioningCombine` · `ConditioningConcat` ·
  `ConditioningAverage`

### Model

- **Checkpoint loaders** (terminus, returns `ckpt_name` / `unet_name`):
  `CheckpointLoaderSimple` · `CheckpointLoader` · `unCLIPCheckpointLoader` ·
  `Checkpoint Loader`
- **Passthrough** (continue upstream): `LoraLoader` · `LoraLoaderModelOnly` ·
  `Power Lora Loader (rgthree)` · `Lora Loader Stack (rgthree)` · `ModelMergeSimple` ·
  `ModelMergeAdd` · `ModelMergeBlocks` · `CLIPSetLastLayer` · `ModelSamplingDiscrete` ·
  `ModelSamplingContinuousEDM` · `FreeU` · `FreeU_V2` · `PerturbedAttentionGuidance` ·
  `SelfAttentionGuidance` · `RescaleCFG` · `DifferentialDiffusion` · `Reroute`

### Image size — latent nodes

Width/height read from `EmptyLatentImage` · `EmptySD3LatentImage` ·
`EmptyHunyuanLatentVideo` (direct or Primitive-referenced scalars).

`Reroute` and any switch node whose class name contains `Switch` (e.g. `ImpactSwitch`,
`ImpactInversedSwitch`) are followed automatically, using the `select` input to pick
the active branch.

## Adding a node

Most gaps are fixed by adding a class name to the right set near the top of
[main.py](main.py) — no new logic needed if the node merely passes a signal through:

- pure model passthrough (e.g. another LoRA loader) → `MODEL_PASSTHROUGH`
- conditioning passthrough / merge → `COND_PASSTHROUGH` / `COND_COMBINE`
- another CLIP text encoder → `TEXT_ENCODE_CLASSES`
- a sampler variant → `SAMPLER_CLASSES`
- an empty-latent variant → `LATENT_SIZE_CLASSES`

Nodes that transform text (concatenate, regex, wildcards, …) need a dedicated branch in
`resolve_text`. PRs welcome.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Mikolas Bingemer.
