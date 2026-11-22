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
workflow graph embedded in each image at scan time. Images it cannot make sense of
(no known sampler node, an error while walking) are handed to IIB's built-in
parser, so the plugin is never worse than not having it.

The code is split in two files:

- `graphwalk.py`: the graph walk itself. Standard library only, no IIB, no PIL, so it
  can be used on its own (see [Using the core without IIB](#using-the-core-without-iib)).
- `main.py`: the IIB adapter. Reads the `prompt` chunk from the image, calls the core
  and hands the result to IIB.

`tests/` holds the unit tests, `tools/compare_versions.py` a regression check
between two versions of the core (see [Tests](#tests)).

## Requirements

- **Python ≥ 3.10** (uses PEP 604 `X | None` type hints)
- A reasonably recent **IIB** with the plugin loader (`plugins/` directory). The plugin
  imports a few IIB internals, including one private helper for a non-standard EXIF
  fallback; that helper's absence degrades only that fallback, not the whole plugin.
  Tested against IIB `main` at commit `9904f6fd` (v1.12.0 plus four commits), run in
  a container; the in-place refresh below needs v1.10.0 or newer.

## Install

1. Copy the whole `comfyui_graphwalk/` folder into IIB's `plugins/` directory. Both
   `main.py` and `graphwalk.py` are needed; `main.py` alone fails to load and IIB
   falls back to its built-in parser.
2. **Restart IIB** (plugins load only at startup). The console prints
   `IIB loaded plugin: ComfyUI Graph-Walk`.

Every **newly scanned** image is now parsed correctly.

## Refresh already-indexed images (one-time)

IIB caches parsed metadata in its SQLite DB (`iib.db`), so already-indexed images keep
their old (wrong) info until they are parsed again. Back up the DB file first, stop
IIB, then run **one** of the following.

### Recommended — re-parse in place (IIB ≥ v1.10.0)

From IIB's root folder (the one containing `app.py`), with IIB's Python environment and
the same `IIB_DB_PATH` IIB uses (default: `iib.db` in that folder; in a container, run
it with the same volumes and environment as IIB):

```bash
python - <<'EOF'
import os
from contextlib import closing
from scripts.iib.db.datamodel import DataBase
from scripts.iib.db.update_image_data import add_image_data_single

conn = DataBase.get_conn()
with closing(conn.cursor()) as cur:
    cur.execute("SELECT path FROM image WHERE exif_edited = 0 AND exif LIKE '%Source Identifier: ComfyUI%'")
    paths = [row[0] for row in cur.fetchall()]
done = 0
for path in paths:
    if not os.path.isfile(path):
        continue
    conn.execute("UPDATE image SET date = '' WHERE path = ?", (path,))
    conn.commit()
    add_image_data_single(path)
    done += 1
print("re-parsed:", done, "of", len(paths))
EOF
```

An empty `date` makes IIB's incremental indexer treat the file as changed. From v1.10.0
on it then parses the image again, removes only the old automatic tags (model, sampler,
LoRA, prompt, …), updates the existing row in place and restores the date from the
file. The row keeps its id, so **custom tags and likes stay**; manually edited entries
(`exif_edited=1`) are not touched. The script can be interrupted and re-run: rows left
with an empty date are picked up by IIB's next scan anyway.

Optionally click **"Rebuild image index"** afterwards: it recomputes the tag counts,
which the in-place re-parse leaves too high for tags that were removed and added again.

Older IIB versions (before v1.10.0) delete the image row when a file looks changed,
and with it its custom tags. Do not use this on them.

### Simplest — full reset

Stop IIB, delete the DB, restart. Re-scans everything; **custom tags and likes are lost**.

```bash
rm ~/sd-webui-infinite-image-browsing/iib.db
```

### Not recommended

- **Deleting the ComfyUI rows** of `image` / `image_tag` with SQL: it removes custom
  tags and likes along with them.
- **`touch`-ing image files** forces a re-scan but destroys original mtimes (breaks
  date sorting).
- **"Rebuild image index" alone** rebuilds tags but does **not** refresh the cached
  `exif`, so detail-panel info stays stale. Pair it with the re-parse above.

## How it works

Each ComfyUI image embeds its full workflow graph. The plugin walks that graph
backwards from the sampler node — through LoRA loaders, Reroutes, conditioning merges
and other intermediates — until it reaches the source values (checkpoint name, CLIP
encoders, latent size, seed/steps/CFG scalars). These are assembled into an A1111-style
`parameters` string, so tags, search and "Send to txt2img" keep working. IIB's
`parse_generation_parameters` turns the prompt part into prompt tags (given a comma
before the LoRA tags, so the last phrase of the prompt stays a tag of its own); the metadata
(`Steps`, `Sampler`, `CFG scale`, `Seed`, `Size`, `Model`, `Source Identifier`) and the
LoRA list come straight from the extracted fields, so they are complete even when only
a few parameters could be resolved and also cover LoRAs IIB's own `<lora:…>` pattern
cannot read (negative weights, names with characters such as `+` or `'`).

Prepended to IIB's parser list, it transparently replaces the built-in ComfyUI parser;
other image types (A1111, Fooocus, …) are untouched. PNG, WEBP and JPEG are supported
(WEBP/JPEG metadata is read like IIB does: structured EXIF first, then the
null-byte and UserComment fallbacks).

When the graph has no known sampler (e.g. `SamplerCustomAdvanced`) or the walk fails,
the image is parsed by IIB's built-in ComfyUI parser instead; if that fails too, only
the image dimensions are stored. Failures are logged as warnings (file path and error,
no prompt text) through IIB's logger.

Coverage is broad but not exhaustive — workflows using node types not listed below may
produce incomplete results. The rule is *better empty than wrong*: where the graph needs
a rule the plugin does not know (an unknown switch, a size it cannot read), the value
stays unresolved instead of being guessed.

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

From the command line, `python3 graphwalk.py <image.png | prompt.json>` prints the
A1111 text of an image (PNG `prompt` chunk, read without Pillow) or of a prompt JSON file.
With `--trace` it prints the path taken for every field instead — node id, class and
input name of each step, the branch a switch selected, the values read — which is what
a bug report needs. `trace_from_prompt(prompt)` returns the same as `(info, lines)`.

`extract_from_prompt(prompt)` returns a dict with `positive`, `negative`, `seed`,
`steps`, `cfg`, `sampler_name`, `scheduler`, `model_name`, `width`, `height` and
`loras` (list of `(name, weight)`); unresolved values are `""`, `None` or `[]`. A field
whose walk fails is reported as unresolved without affecting the others; input that is
not a prompt dict gives `None`. `format_a1111(info)` turns it into an A1111-style
`parameters` string, LoRAs appended to the positive prompt as `<lora:name:weight>`
tags. The parts it is made of are available too: `a1111_prompt_text(info)`,
`a1111_params(info)` (ordered `(key, value)` strings), `lora_entries(info)` and
`model_basename(name)`. Import `graphwalk`, not `main`: `main.py` needs IIB.

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

The seed is read from `noise_seed` where the sampler has that input, else from `seed`.

The Efficient samplers are recognised, but their prompts, size and LoRAs usually sit
on the `Efficient Loader` (string widgets for positive/negative, empty-latent size,
LoRA widgets), which is not read: with that loader, only the checkpoint name,
seed, steps, CFG, sampler and scheduler come out. Without a sampler the image goes to
IIB's built-in parser (see above).

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
| `StringFunction` / `StringFunction\|pysssss` | as the node's own code: `append` joins the non-empty `text_a`, `text_b`, `text_c` with `", "` when `tidy_tags` is `yes`, else with nothing; `replace` replaces `text_b` in `text_a` by `text_c` (`/regex/` supported); `tidy_tags` collapses blanks and repeated commas |
| `StringConcatenate` | `string_a` + `delimiter` + `string_b` as in core (delimiter default empty, empty parts kept) |
| `RegexReplace` | follows upstream text, then applies the substitution; pattern, replacement, count and flags may be wired from other nodes |
| Nodes in `TEXT_PASSTHROUGH` | output one of their own text inputs unchanged; the map (class → input name) at the top of [graphwalk.py](graphwalk.py) names which input is read |
| Switches | see [Switches](#switches) |
| Display passthrough — `easy showAnything` · `ShowText\|pysssss` · `Display Any (rgthree)` / … | follows the passthrough input; the cached `text`/`value`/`string` widget is the fallback when the source cannot be resolved |
| Primitive / `ShowText` / `Note` / `TriggerWord Toggle` / … | any node storing a scalar in `value`, `text`, `string`, `populated_text`, or `trigger_words` |

### Conditioning

- **Passthrough** (continue upstream): `Reroute` · `ConditioningSetTimestepRange` ·
  `ConditioningSetArea` · `ConditioningSetAreaPercentage` · `ConditioningSetMask` ·
  `unCLIPConditioning`. An unknown node with a `conditioning` input is followed the same
  way (marked "generic" in the trace); one with several conditioning inputs is not.
- **Merge** (all inputs joined): `ConditioningCombine` · `ConditioningConcat` ·
  `ConditioningAverage`
- **Slot-aware passthrough**, one rule for every node that has both a `positive` and a
  `negative` input (`ControlNetApplyAdvanced`, `ControlNetApplySD3`,
  `InstructPix2PixConditioning`, `WanImageToVideo`, …): output slot 0 comes from
  `positive`, slot 1 from `negative`, any later slot is not conditioning. The deprecated
  single-input `ControlNetApply` follows its `conditioning` input
- **`ConditioningZeroOut`** yields no text: an empty negative prompt stays empty
  instead of repeating the positive one

### Switches

Followed on text, conditioning, model, latent and scalar edges. Only the switches in the
`SWITCHES` table of [graphwalk.py](graphwalk.py) are known; a node that merely has
"Switch" in its name is not guessed at (its inputs stay unresolved).

| Node | Rule |
|---|---|
| core `ComfySwitchNode` / `ComfySoftSwitchNode` | `on_true` / `on_false` by `switch` (a literal, or wired from a scalar node). The soft variant uses the connected branch when only one is connected. A flag that cannot be resolved selects nothing |
| rgthree `Any Switch` | the first connected `any_*` input, by number |
| Impact `ImpactSwitch`, `LatentSwitch` | `input<select>` (`select` literal or wired). No such input (or no `select`) gives nothing, as the node itself outputs nothing then |
| Impact `ImpactInversedSwitch` | one `input`, output slot `select - 1` carries it, the other outputs nothing |
| Easy-Use `easy anythingIndexSwitch` / `textIndexSwitch` / `conditioningIndexSwitch` | `value<index>` / `text<index>` / `cond<index>`, index from 0 |
| Easy-Use `easy textSwitch` | `text<input>` |
| Easy-Use `easy anythingInversedSwitch` | one `in`, output slot `index` carries it |

### Sampler scalars — seed / steps / CFG / sampler / scheduler

Read from the sampler's own widgets, or resolved along graph edges when wired from
upstream nodes. The resolver carries the *name* of the field being resolved so it can:

- follow [switch](#switches) nodes (selected branch), `Reroute` and `Any to String
  (Image Saver)`, and read plain value nodes (`Primitive*`, `easy int` / `easy float`,
  …) from their `value` / `int` / `float` / `number` / `result` widget. A `seed` widget
  (`Seed (rgthree)`) answers only a request for the seed, so `steps` never comes from a
  seed; string fields (sampler, scheduler, file names) read only `value`;
- map the output slot of a multi-output node to the widget behind it (`OUTPUT_WIDGETS`):
  `Input Parameters (Image Saver)` (seed, steps, cfg, sampler, scheduler, denoise), the
  Image Saver sampler / scheduler selectors and literals;
- as the rule of last resort, match a widget named like the requested field on an
  aggregator / pipe node;
- resolve display nodes (`easy showAnything`, `ShowText`, `PreviewAny`, …) by following
  their input edge. Their cached widget is a leftover of an earlier run and goes stale
  when the source is edited, so it is read only when the source cannot be resolved.

### Model

- **Loaders** (terminus; the `MODEL_LOADERS` table names the widget, `ckpt_name` or
  `unet_name`, literal or wired from another node): `CheckpointLoaderSimple` ·
  `CheckpointLoader` · `unCLIPCheckpointLoader` · `ImageOnlyCheckpointLoader` · `UNETLoader` ·
  the GGUF loaders · the LoraManager and Image Saver loaders and others. A node of an
  unknown class at the end of the chain (no `model` input) is read through `ckpt_name` /
  `unet_name`; other widgets such as `model_name` (upscalers, detectors) do not end it.
- **Passthrough** (continue upstream): **any** node carrying a `model` edge is followed
  generically — no whitelist. This covers LoRA loaders, model patchers
  (`CLIPSetLastLayer`, `ModelSamplingDiscrete`, `FreeU`, `RescaleCFG`,
  `DifferentialDiffusion`, …), CLIP-conditioning patchers (`CLIPNegPip`), IPAdapter
  variants and `Reroute`, plus any future node that merely forwards the model. Switch
  nodes on the model edge follow their selected branch.
- **Model merges** (`ModelMergeSimple`, `ModelMergeAdd`, `ModelMergeBlocks`, … with
  `model1` / `model2`): the model name is that of `model1`; LoRAs are collected from
  both inputs.
- The shown model name is the file name without folder (`/` or `\`) and extension.

### LoRAs

The model chain is also walked to collect every applied LoRA, which is emitted as
A1111-style `<lora:name:weight>` tags appended to the positive prompt. The adapter
also hands the same list to IIB's indexer, which turns it into searchable `lora` tags
(tag search, filters, LoRA statistics) the same way as for A1111 images. IIB's own
tag pattern cannot read negative weights (slider LoRAs) or names with characters such
as `+` or `'`; the list is built from the graph, so those get a tag too. ComfyUI keeps
LoRAs in loader nodes rather than in the prompt text, so without this they are
invisible. LoRA-loader nodes understood:

- `Power Lora Loader (rgthree)` (per-slot `lora_N` = `{on, lora, strength}`, only
  slots that are `on`) · `LoraLoader` · `LoraLoaderModelOnly` (skipped when
  `strength_model` and `strength_clip` are both 0, i.e. switched off; strengths may be
  wired from other nodes) · `Lora Loader Stack (rgthree)` (`None` = empty slot)
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
the size is read from the first `W×H` / `WxH` in one of its literal string widgets (e.g.
a preset label `Portrait 2:3 · 1152×1728 · 1.9 MP`, width first); without one, a widget
whose name contains `width` / `height` (e.g. `custom_width`) is used. This applies only
to the node at the end of the `width` / `height` edge of an empty-latent node, never to
other fields. Pickers that compute the size from aspect ratio and megapixels are not
covered.

`Reroute` and the [switch nodes](#switches) are followed automatically.

## Adding a node

The walker is generic; what a node does is data in the tables near the top of
[graphwalk.py](graphwalk.py). Most gaps are fixed by adding a class name to the right one:

- pure model passthrough (e.g. another LoRA loader or a merge) → already handled (any
  `model` / `model1` / `model2` edge is followed generically); no change needed
- a model loader → `MODEL_LOADERS` (class → file-name widget)
- conditioning passthrough / merge → `COND_PASSTHROUGH` / `COND_COMBINE`; a node with
  `positive` and `negative` inputs needs nothing
- another CLIP text encoder → `TEXT_ENCODE_CLASSES`
- a switch → `SWITCHES` (a `Switch` entry says how it picks its input)
- a multi-output node whose widgets are the values → `OUTPUT_WIDGETS` (slot → widget)
- a node that passes a value through unchanged → `TEXT_PASSTHROUGH` (text input name) or
  `VALUE_PASSTHROUGH`
- a display node with a cached widget → `SHOW_PASSTHROUGH`
- a sampler variant → `SAMPLER_CLASSES`
- an empty-latent variant → `LATENT_SIZE_CLASSES`

Nodes that transform text (concatenate, regex, wildcards, …) need a dedicated branch in
`Walker._text_of`. PRs welcome.

## Tests

Core tests, standard library only:

```bash
python3 -m unittest discover tests
```

`tests/test_main.py` tests the IIB adapter and is skipped without IIB. Run it inside
an IIB checkout or container, e.g. with the Docker image of IIB:

```bash
docker run --rm -v "$PWD":/app/plugins/comfyui_graphwalk:ro -w /app <iib-image> \
    python -m unittest discover -s /app/plugins/comfyui_graphwalk/tests
```

All graphs and prompts in the tests are synthetic: `test_graphwalk.py` the behaviour of the
whole extraction, `test_rules.py` one test per rule table and per heuristic that used to
guess, `test_trace.py` the trace mode and the command line. To check that a change leaves the
output for your own images unchanged, compare two versions of the core with
`tools/compare_versions.py` (Pillow required). It prints only counts, file paths and
the names of differing fields, never prompt text:

```bash
git show main:graphwalk.py > /tmp/graphwalk_old.py
python3 tools/compare_versions.py /tmp/graphwalk_old.py graphwalk.py /path/to/images
```

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Mikolas Bingemer.
