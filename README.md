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
- `main.py`: the IIB adapter. Reads the `prompt` chunk (and, for PNG, resources from
  an Image Saver `parameters` chunk) from the image, calls the core and hands the
  result to IIB.

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

Each ComfyUI image embeds its full workflow graph. The plugin finds the sampler that
made the saved picture (from the save node back to the first pass, see
[Which sampler is read](#which-sampler-is-read)) and walks that graph
backwards from it — through LoRA loaders, Reroutes, conditioning merges
and other intermediates — until it reaches the source values (checkpoint name, CLIP
encoders, latent size, seed/steps/CFG scalars). These are assembled into an A1111-style
`parameters` string, so tags, search and "Send to txt2img" keep working. The scheduler
is its own key, as in A1111 (`Sampler: euler, Schedule type: simple`). IIB's
`parse_prompt` turns the prompt text into prompt tags (given a comma before the LoRA
tags, so the last phrase of the prompt stays a tag of its own; IIB versions without
`parse_prompt` get the same tags through `parse_generation_parameters`); the metadata
(`Steps`, `Sampler`, `Schedule type`, `CFG scale`, `Seed`, `Size`, `Model`, `Source Identifier`, plus [Image Saver resources](#image-saver-resources) when present) and the
LoRA list come straight from the extracted fields, so they are complete even when only
a few parameters could be resolved and also cover LoRAs IIB's own `<lora:…>` pattern
cannot read (negative weights, names with characters such as `+` or `'`).

Prepended to IIB's parser list, it transparently replaces the built-in ComfyUI parser;
other image types (A1111, Fooocus, …) are untouched. PNG, WEBP and JPEG are supported
(WEBP/JPEG metadata is read like IIB does: structured EXIF first, then the
null-byte and UserComment fallbacks).

When the graph has no sampler graphwalk knows (see [Which sampler is read](#which-sampler-is-read))
or the walk fails, the image is parsed by IIB's built-in ComfyUI parser instead; if that fails too, only
the image dimensions are stored. Failures are logged as warnings (file path and error,
no prompt text) through IIB's logger.

Coverage is broad but not exhaustive — workflows using node types not listed below may
produce incomplete results. The rule is *better empty than wrong*: where the graph needs
a rule the plugin does not know (an unknown switch, a size it cannot read), the value
stays unresolved instead of being guessed.

### Image Saver resources

PNGs saved by the Image Saver node (comfyui-image-saver) carry an A1111 `parameters`
text chunk next to the `prompt` graph. The graph stays the source of everything it
holds (prompts, sampler, seed, …; the chunk may spell them in Civitai's names). Only the resources are taken from the chunk, so Civitai and hash
lookups keep working:

| `meta` key (string) | `extra` key | Content |
|---|---|---|
| `Model hash` | `model_hash` | checkpoint hash |
| `Hashes` | `hashes` | dict: `model`, `LORA:<name>`, `embed:<name>` → hash |
| `Lora hashes` | `lora_hashes` | dict: LoRA name → hash |
| `Civitai resources` | `civitai_resources` | list of the resource objects |

Both Image Saver forms are read (older versions write only `Civitai resources`). The
chunk is parsed here rather than by IIB's regex, which splits unquoted JSON at its
commas. In the A1111 text they sit in the settings line around `Model`
(`…, Model hash: <hash>, Model: <name>, Hashes: "<json>", Lora hashes: "<list>",
Civitai resources: "<json>"`); the three with commas are JSON-quoted so that IIB's
parser and the web UI read them whole. They are not turned into tags. A missing or
malformed chunk changes nothing. Only PNG is read.

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
`a1111_params(info)` (ordered `(key, value)` strings, the scheduler as `Schedule type`),
`lora_entries(info)` and `model_basename(name)`. `image_saver_resources(parameters_text)`
reads the [Image Saver resources](#image-saver-resources) from an A1111 text; passed as
second argument to `format_a1111` and `a1111_params` they are added to the output. Import `graphwalk`, not `main`: `main.py` needs IIB.

## Supported nodes

### Which sampler is read

A graph can hold several samplers (hires fix, refiner, detailer). The one read is the one
that made the saved picture, found like this:

1. **Output nodes.** Save nodes are the start: the known classes (`SaveImage`,
   `SaveImageAdvanced`, `SaveAnimatedWEBP` / `PNG`, `SaveWEBM`, `SaveVideo`, `Image Saver`,
   `Image Saver Simple`, `Image Saver (From Pipe)`, `Save Image (LoraManager)`,
   `SaveImageKJ`, `SaveImageWithAlpha`, `easy imageSave`, `SaveImageNaming`; table
   `SAVE_CLASSES`), then any other node with a connected `images` input that no other node
   uses (a video combiner, say). Preview nodes (`PreviewImage` and the like,
   `PREVIEW_CLASSES`) count only when no save node leads to a sampler: IIB indexes saved images.
2. **Back to the latent.** From the output node the picture is followed backwards: through
   image nodes (`image` / `images` / `pixels` / `video`: upscalers, detailers, `CreateVideo`),
   `VAEDecode` / `VAEDecodeTiled` (`samples`), `VAEEncode` (`pixels`) and latent nodes, until a
   node has nothing left to follow, normally an empty latent.
3. **First pass.** Of the samplers on that way the first pass counts, as in A1111's hires
   fix: the one whose latent comes from an empty-latent node. Without such a start (img2img:
   the chain begins at a loaded image) the sampler nearest the output counts. The size is
   that of the empty latent.
4. **Several output nodes** are tried by node number (`9` before `12`, not by text), the
   first with a sampler behind it decides.
5. **Fallback.** If no output node leads to a sampler, the first `KSampler` /
   `KSamplerAdvanced` in the graph is used, then the other known classes, then the first
   node (by number) that looks like a sampler. A graph that has a sampler always gives a
   result.

`--trace` shows the output node tried, the path back, and the sampler chosen with the
reason (first pass, nearest to the output, or fallback).

### Sampler entry points

| Node | Pack |
|---|---|
| `KSampler` | built-in |
| `KSamplerAdvanced` | built-in |
| `KSampler (Efficient)` | efficiency-nodes |
| `KSamplerAdvanced (Efficient)` | efficiency-nodes |
| `Tiled KSampler` | ComfyUI-TiledDiffusion |
| `BNK_Unsampler` | ComfyUI-BNK |
| `SamplerCustom` | built-in |
| `SamplerCustomAdvanced` | built-in |

Any other node counts as a sampler by its inputs: `model`, `positive`, `negative`,
`latent_image`, `steps` and a `seed` or `noise_seed` (KSampler-like), or the inputs of
`SamplerCustom` (`sampler`, `sigmas`, `noise_seed`, …) or `SamplerCustomAdvanced` (`noise`,
`guider`, `sampler`, `sigmas`, `latent_image`).

The seed is read from `noise_seed` where the sampler has that input, else from `seed`.

The Efficient samplers are recognised, but their prompts, size and LoRAs usually sit
on the `Efficient Loader` (string widgets for positive/negative, empty-latent size,
LoRA widgets), which is not read: with that loader, only the checkpoint name,
seed, steps, CFG, sampler and scheduler come out. Without a sampler the image goes to
IIB's built-in parser (see above).

### Custom samplers — guider, noise, sampler, sigmas

`SamplerCustom` and `SamplerCustomAdvanced` take most of their settings from nodes wired
to them. Each of these edges is followed like any other (`Reroute` and
[switches](#switches) included) to the node it comes from; a node that is not listed is
not read, and its fields stay empty.

| Input | Nodes | Read |
|---|---|---|
| `guider` | `CFGGuider` · `DualModelGuider` · `PerpNegGuider` · `ScheduledCFGGuidance` | `model`, `positive`, `negative`, `cfg` |
| | `BasicGuider` | `model`, `conditioning` as the positive prompt; no negative, no CFG |
| | `DualCFGGuider` | `model`, `cond1` as the positive prompt, `negative`; no CFG (there are two) |
| | `LTXVDualCFGGuider` | `model`, `positive`, `negative`, `video_cfg` as the CFG |
| `noise` | `RandomNoise` | `noise_seed` |
| | `DisableNoise` | no seed (`SamplerCustom` with `add_noise` off likewise) |
| `sampler` | `KSamplerSelect` | `sampler_name` |
| | `SamplerEulerAncestral` · `SamplerEulerAncestralCFGPP` · `SamplerDPMPP_2S_Ancestral` · `SamplerLMS` · `SamplerDPMAdaptative` · `SamplerER_SDE` | the sampler the node stands for (`euler_ancestral`, `lms`, …) |
| | `SamplerDPMPP_3M_SDE` · `SamplerDPMPP_2M_SDE` · `SamplerDPMPP_SDE` | `dpmpp_3m_sde` etc., with `_gpu` unless `noise_device` is `cpu`, as the nodes do |
| `sigmas` | `BasicScheduler` | `scheduler` (widget), `steps`; its `model` if the guider is not known |
| | `KarrasScheduler` · `ExponentialScheduler` · `PolyexponentialScheduler` · `LaplaceScheduler` · `BetaSamplingScheduler` · `VPScheduler` · `SDTurboScheduler` · `AlignYourStepsScheduler` · `GITSScheduler` · `OptimalStepsScheduler` · `Flux2Scheduler` · `LTXVScheduler` · `Ideogram4Scheduler` | `steps`; the scheduler is named after the node: `karras`, `exponential`, `beta` as in ComfyUI's own list, else `polyexponential`, `laplace`, `vp`, `sd_turbo`, `align_your_steps`, `gits`, `optimal_steps`, `flux2`, `ltxv`, `ideogram4` |
| | `FlipSigmas` · `SetFirstSigma` | passed through, steps unchanged |
| | `SplitSigmas` · `SplitSigmasDenoise` · `ExtendIntermediateSigmas` | passed through; the number of steps is no longer the scheduler's, so `steps` stays empty |

The model, and so the checkpoint name and the LoRAs, comes from the guider's `model`
(through `ModelSampling*`, `CFGNorm`, LoRA loaders and the like, as for a KSampler); the
latent and so the size from `latent_image`. `cfg` is `None` for `BasicGuider` (Flux's
guidance is not a CFG scale and is not extracted, see [Conditioning](#conditioning)).

### Prompt text — CLIP encoders

Reached by following the `positive` / `negative` conditioning edges (of the sampler, or of
its guider). Encoders with several text inputs give all of them, in the order below; equal
texts count once, different ones are joined with a line break.

| Node | Text inputs |
|---|---|
| `CLIPTextEncode` | `text`: standard SD 1.x / 2.x |
| `CLIPTextEncodeSDXL` | `text_g` (global), `text_l` (local) |
| `CLIPTextEncodeSDXLRefiner` | `text` |
| `smZ CLIPTextEncode` | smZNodes: `text_g`, `text_l`, `text` |
| `BNK_CLIPTextEncodeAdvanced` | ComfyUI-BNK: `text_g`, `text_l`, `text` |
| `CLIPTextEncodeFlux` | `clip_l`, `t5xxl` |
| `CLIPTextEncodeSD3` | `clip_l`, `clip_g`, `t5xxl` |
| `CLIPTextEncodeHiDream` | `clip_l`, `clip_g`, `t5xxl`, `llama` |
| `CLIPTextEncodeHunyuanDiT` | `bert`, `mt5xl` |
| `CLIPTextEncodeKandinsky5` | `clip_l`, `qwen25_7b` |
| `CLIPTextEncodeLumina2` | `user_prompt` (the system prompt is a preset) |
| `CLIPTextEncodePixArtAlpha` | `text` |
| `TextEncodeQwenImageEdit` · `TextEncodeQwenImageEditPlus` | `prompt` |
| `TextEncodeHunyuanVideo_ImageToVideo` | `prompt` |

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
  `ConditioningSetArea` · `ConditioningSetAreaPercentage` · `ConditioningSetAreaStrength` ·
  `ConditioningSetMask` · `unCLIPConditioning` · `FluxGuidance` · `ReferenceLatent` ·
  `FluxKontextMultiReferenceLatentMethod`. `FluxGuidance` only sets a `guidance` value on the
  conditioning; it is not part of the result. An unknown node with a `conditioning` input is followed the same
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

Width/height read from `EmptyLatentImage` · `EmptySD3LatentImage` · `EmptyFlux2LatentImage` ·
`EmptyHunyuanLatentVideo` · `EmptyHunyuanVideo15Latent` · `EmptyHunyuanImageLatent` ·
`EmptyQwenImageLayeredLatentImage` · `EmptyLTXVLatentVideo` · `EmptyMochiLatentVideo` ·
`EmptyCosmosLatentVideo` · `EmptyChromaRadianceLatentImage` · `EmptyHiDreamO1LatentImage` ·
`EmptyARVideoLatent` · `EmptyMiniMaxH3LatentAV` (direct or Primitive-referenced scalars).
Any other node whose class name starts with `Empty`, contains `Latent` and that has `width`
and `height` inputs counts the same way. For the choice of the [first pass](#which-sampler-is-read)
any `Empty…Latent…` node is a start, with or without a size.

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
- another CLIP text encoder → `TEXT_ENCODERS` (class → its text inputs, in order)
- a switch → `SWITCHES` (a `Switch` entry says how it picks its input)
- a multi-output node whose widgets are the values → `OUTPUT_WIDGETS` (slot → widget)
- a node that passes a value through unchanged → `TEXT_PASSTHROUGH` (text input name) or
  `VALUE_PASSTHROUGH`
- a display node with a cached widget → `SHOW_PASSTHROUGH`
- a sampler variant → `SAMPLER_CLASSES` (not needed if it has a KSampler's inputs)
- a guider, noise, sampler or scheduler node of `SamplerCustomAdvanced` → `GUIDERS`,
  `NOISE_SEEDS`, `SAMPLER_NODES`, `SCHEDULERS` (and `SIGMAS_PASSTHROUGH` for nodes that
  take sigmas and give sigmas)
- a node that saves pictures → `SAVE_CLASSES` (class → the input taking the pictures; one
  with an `images` input that nothing uses is found without it)
- an empty-latent variant → `LATENT_SIZE_CLASSES` (not needed if it is named `Empty…Latent…`
  and has `width` and `height`)

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
guess, `test_sampler.py` the choice of the sampler (hires fix, refiner, img2img, several
output nodes, fallback), `test_custom_sampler.py` `SamplerCustom(Advanced)` with every
guider, noise, sampler and scheduler node, `test_models.py` the encoders and empty latents
of newer models, `test_trace.py` the trace mode and the command line. To check that a change leaves the
output for your own images unchanged, compare two versions of the core with
`tools/compare_versions.py` (Pillow required). It prints only counts, file paths and
the names of differing fields, never prompt text:

```bash
git show main:graphwalk.py > /tmp/graphwalk_old.py
python3 tools/compare_versions.py /tmp/graphwalk_old.py graphwalk.py /path/to/images
```

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Mikolas Bingemer.
