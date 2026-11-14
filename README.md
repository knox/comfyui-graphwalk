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

The code is split in two files:

- `graphwalk.py`: the graph walk itself. Standard library only, no IIB, no PIL, so it
  can be used on its own (see [Using the core without IIB](#using-the-core-without-iib)).
- `main.py`: the IIB adapter. Reads the `prompt` chunk from the image, calls the core
  and hands the result to IIB.

## Requirements

- **Python ≥ 3.10** (uses PEP 604 `X | None` type hints)
- A reasonably recent **IIB** with the plugin loader (`plugins/` directory). The plugin
  imports a few IIB internals, including one private helper for a non-standard EXIF
  fallback; that helper's absence degrades only that fallback, not the whole plugin.
  Tested against [IIB v1.10.1](https://github.com/zanllp/infinite-image-browsing/releases/tag/v1.10.1).

## Install

1. Copy the whole `comfyui_graphwalk/` folder into IIB's `plugins/` directory. Both
   `main.py` and `graphwalk.py` are needed; `main.py` alone fails to load and IIB
   falls back to its built-in parser.
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

## Using the core without IIB

`graphwalk.py` takes the already parsed ComfyUI `prompt` JSON; reading it from the file
is up to the caller. Example with Pillow for a PNG:

```python
import json
import sys

from PIL import Image

sys.path.insert(0, "/path/to/comfyui_graphwalk")
import graphwalk

with Image.open("ComfyUI_00001_.png") as img:
    prompt = json.loads(img.info["prompt"])

info = graphwalk.extract_from_prompt(prompt)  # None if the graph has no sampler
if info:
    print(info["positive"], info["model_name"], info["loras"])
    print(graphwalk.format_a1111(info))
```

`extract_from_prompt(prompt)` returns a dict with `positive`, `negative`, `seed`,
`steps`, `cfg`, `sampler_name`, `scheduler`, `model_name`, `width`, `height` and
`loras` (list of `(name, weight)`); unresolved values are `""`, `None` or `[]`.
`format_a1111(info)` turns it into an A1111-style `parameters` string, LoRAs appended
to the positive prompt as `<lora:name:weight>` tags. Import `graphwalk`, not `main`:
`main.py` needs IIB.

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
| `StringConcatenate` | joins `string_a` and `string_b` using the `delimiter` widget (default `" "`) |
| Nodes in `TEXT_PASSTHROUGH` | output one of their own text inputs unchanged; the map (class → input name) at the top of [graphwalk.py](graphwalk.py) names which input is read |
| Any `*Switch` node | follows the branch chosen by `select` (falls back to the first connected input) |
| Display passthrough — `easy showAnything` · `ShowText\|pysssss` · `Display Any (rgthree)` / … | reads the cached `text`/`value`/`string` widget, else follows the passthrough input |
| Primitive / `ShowText` / `Note` / `TriggerWord Toggle` / … | any node storing a scalar in `value`, `text`, `string`, `populated_text`, or `trigger_words` |

### Conditioning

- **Passthrough** (continue upstream): `Reroute` · `ConditioningSetTimestepRange` ·
  `ConditioningSetArea` · `ConditioningSetAreaPercentage` · `ConditioningSetMask` ·
  `unCLIPConditioning`
- **Merge** (both inputs joined): `ConditioningCombine` · `ConditioningConcat` ·
  `ConditioningAverage`
- **Slot-aware passthrough** (output slot 0 → `positive` input, slot 1 → `negative` input):
  `ControlNetApplyAdvanced` · `ControlNetApply`

### Sampler scalars — seed / steps / CFG / sampler / scheduler

Read from the sampler's own widgets, or resolved along graph edges when wired from
upstream nodes. The resolver carries the *name* of the field being resolved so it can:

- follow `*Switch` nodes (selected branch) and Primitive / `Seed (rgthree)` / `easy int`
  / `easy float` scalar sources;
- match a widget on the target node that is **named like the requested field** — this
  handles aggregator / pipe nodes such as `Input Parameters (Image Saver)`, where
  `KSampler.steps` points at an output slot but the value lives in a widget literally
  called `steps` (likewise `cfg`, `seed`, …);
- read the cached widget of a display passthrough (`easy showAnything`, `ShowText`, …),
  e.g. a `sampler_name` routed through a preview node.

### Model

- **Checkpoint loaders** (terminus, returns `ckpt_name` / `unet_name` / `model_name`):
  `CheckpointLoaderSimple` · `CheckpointLoader` · `unCLIPCheckpointLoader` ·
  `Checkpoint Loader`
- **Passthrough** (continue upstream): **any** node carrying a `model` edge is followed
  generically — no whitelist. This covers LoRA loaders, model merges, model patchers
  (`CLIPSetLastLayer`, `ModelSamplingDiscrete`, `FreeU`, `RescaleCFG`,
  `DifferentialDiffusion`, …), CLIP-conditioning patchers (`CLIPNegPip`), IPAdapter
  variants and `Reroute`, plus any future node that merely forwards the model. Switch
  nodes on the model edge follow their selected branch. `MODEL_PASSTHROUGH` in
  [graphwalk.py](graphwalk.py) remains only as a reference list of commonly-seen classes.

### LoRAs

The model chain is also walked to collect every applied LoRA, which is emitted as
A1111-style `<lora:name:weight>` tags appended to the positive prompt. IIB's backend
turns these tags into searchable `lora` tags (tag search, filters, LoRA statistics),
the same way it does for A1111 images; ComfyUI keeps LoRAs in loader nodes rather
than in the prompt text, so without this they are invisible. LoRA-loader nodes
understood:

- `Power Lora Loader (rgthree)` (per-slot `lora_N` = `{on, lora, strength}`, only
  slots that are `on`) · `LoraLoader` · `LoraLoaderModelOnly` ·
  `Lora Loader Stack (rgthree)` (`None` = empty slot)
- `Lora Loader (LoraManager)`: like the node itself, only entries of its `loras`
  widget with `active: true` (also in the `{"__value__": [...]}` form), plus LoRAs
  from a `lora_stack` chain of `Lora Stacker (LoraManager)` nodes. Its `text` widget
  is only an editor mirror that still lists deactivated LoRAs; it is read only when
  the `loras` widget is missing.
- `LoRA Text Loader (LoraManager)`: `<lora:name:w>` / `<lora:name:w:clip_w>` tags of
  `lora_syntax` (literal or linked), plus its `lora_stack`.

Only LoRAs on the sampler's model chain count: loaders behind an unselected switch
branch or off the chain are ignored. Bypassed or muted nodes never reach the API
prompt, so they need no handling. Names are reduced to the file name without folder
and weight-file extension and de-duplicated; IIB itself turns `_` and `-` into spaces
for the tag. Embeddings are not a resource type in IIB; they remain inline in the
prompt as `embedding:name` (the A1111 form).

### Image size — latent nodes

Width/height read from `EmptyLatentImage` · `EmptySD3LatentImage` ·
`EmptyHunyuanLatentVideo` (direct or Primitive-referenced scalars).

When width/height are wired from a size-picker node without `width`/`height` widgets,
the size is read from the first `W×H` / `WxH` in one of its string widgets (e.g. a
preset label `Portrait 2:3 · 1152×1728 · 1.9 MP`); without one, a widget whose name
contains `width` / `height` (e.g. `custom_width`) is used. Pickers that compute the
size from aspect ratio and megapixels are not covered.

`Reroute` and any switch node whose class name contains `Switch` (e.g. `ImpactSwitch`,
`ImpactInversedSwitch`) are followed automatically, using the `select` input to pick
the active branch.

## Adding a node

Most gaps are fixed by adding a class name to the right set near the top of
[graphwalk.py](graphwalk.py) — no new logic needed if the node merely passes a signal through:

- pure model passthrough (e.g. another LoRA loader) → already handled (any `model`
  edge is followed generically); no change needed
- conditioning passthrough / merge → `COND_PASSTHROUGH` / `COND_COMBINE`
- another CLIP text encoder → `TEXT_ENCODE_CLASSES`
- a sampler variant → `SAMPLER_CLASSES`
- an empty-latent variant → `LATENT_SIZE_CLASSES`
- a node that outputs one of its text inputs unchanged → `TEXT_PASSTHROUGH`
  (class → input name)

Nodes that transform text (concatenate, regex, wildcards, …) need a dedicated branch in
`resolve_text`. PRs welcome.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Mikolas Bingemer.
