# ComfyUI Graph-Walk — IIB parser plugin

A drop-in parser plugin for
[sd-webui-infinite-image-browsing](https://github.com/zanllp/sd-webui-infinite-image-browsing)
(IIB) that correctly reads generation metadata from **ComfyUI** images.

## Why

IIB's built-in ComfyUI parser only handles simple, linear workflows and fails silently
on anything more complex, producing wrong or missing model name, prompts, seed, steps,
CFG scale and image size.

This plugin replaces it. It modifies no core IIB files (uninstall by deleting this
folder) and never writes to your image files: all metadata is read from the workflow
graph embedded in each image when IIB scans it. Images it cannot make sense of (no
sampler found, an error while parsing) are handed to IIB's built-in parser, so the
plugin is never worse than not having it.

The code is split in two files:

- `graphwalk.py`: the graph walk itself. Standard library only, no IIB, no PIL, so it
  can be used on its own (see [Using the core without IIB](#using-the-core-without-iib)).
- `main.py`: the IIB adapter. Reads the `prompt` chunk (and, for PNG, resources from
  an Image Saver `parameters` chunk) from the image, calls the core and hands the
  result to IIB.

`tests/` holds the unit tests, `tools/` two helpers: `compare_versions.py`, a
regression check between two versions of the core, and `supported_nodes.py`, which
generates the list of [supported nodes](#supported-nodes) (see [Tests](#tests)).

## Requirements

- **Python ≥ 3.10** (the code uses `X | None` annotations, evaluated at run time)
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
the info they were indexed with until they are parsed again. Do this once after
installing, and again after an update that changes the output; the
[changelog](CHANGELOG.md) says which. Back up the DB file first, stop
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
made the saved picture (see [Which sampler is read](#which-sampler-is-read)) and walks
the graph backwards from it, through LoRA loaders, Reroutes, switches, conditioning
merges and other intermediates, until it reaches the source values: checkpoint name,
text encoders, latent size, seed / steps / CFG scalars. From these it builds what IIB
gets:

- **The `parameters` text** in A1111 style: the prompt with `<lora:name:weight>` tags, a
  `Negative prompt:` line and the settings line (`Steps`, `Sampler`, `Schedule type`,
  `CFG scale`, `Seed`, `Size`, `Model`, and the [Image Saver
  resources](#image-saver-resources) when present). The scheduler is its own key, as in
  A1111: `Sampler: euler, Schedule type: simple`. It is written only together with a
  sampler name.
- **The metadata** (`meta`), taken from the extracted fields rather than parsed back
  from that text: the same keys (`Size` split into `Size-1` / `Size-2`, plus
  `Source Identifier`), complete even when only a few parameters could be resolved.
- **The prompt tags**, from IIB's `parse_prompt` on the prompt text. IIB splits tags at
  commas only, so the LoRA tags get a comma in front and the last phrase of the prompt
  stays a tag of its own. IIB versions without `parse_prompt` get the same tags through
  `parse_generation_parameters`.
- **The LoRA list**, built from the graph. It also covers LoRAs IIB's own `<lora:…>`
  pattern cannot read (negative weights, names with characters such as `+` or `'`).

Prepended to IIB's parser list, the plugin transparently replaces the built-in ComfyUI
parser; other image types (A1111, Fooocus, …) are untouched. PNG, WEBP and JPEG are
supported (WEBP / JPEG metadata is read like IIB does: structured EXIF first, then the
null-byte and UserComment fallbacks).

When the graph has no sampler or parsing raises, the image is parsed by IIB's built-in
ComfyUI parser instead; if that fails too, only the image dimensions are stored. An
exception is logged as a warning (file path and error, no prompt text) through IIB's
logger, a graph without a sampler as info.

Coverage is broad but not exhaustive: workflows using node types not listed under
[Supported nodes](#supported-nodes) may produce incomplete results. The rule is *better
empty than wrong*: where the graph needs a rule the plugin does not know (an unknown
switch, a size it cannot read), the value stays unresolved instead of being guessed.

### Which sampler is read

A graph can hold several samplers (hires fix, refiner, detailer). The one read is the one
that made the saved picture, found like this:

1. **Output nodes.** Save nodes are the start (the classes in the [node
   lists](#node-lists)), then any other node with a connected `images` input that no
   other node uses (a video combiner, say). Preview nodes count only when none of these
   leads to a sampler: IIB indexes saved images.
2. **Back to the latent.** From the output node the picture is followed backwards:
   through image nodes (upscalers, detailers, `CreateVideo`), `VAEDecode` /
   `VAEDecodeTiled` (`samples`), `VAEEncode` (`pixels`) and latent nodes, until a node
   has nothing left to follow, normally an empty latent.
3. **First pass.** Of the samplers on that way the first pass counts, as in A1111's hires
   fix: the one whose latent comes from an empty-latent node. Without such a start
   (img2img: the chain begins at a loaded image) the sampler nearest the output counts.
   The size is that of the empty latent; without one it stays unresolved.
4. **Several output nodes** are tried by node number (`9` before `12`, not by text); the
   first with a sampler behind it decides.
5. **Fallback.** If no output node leads to a sampler: the first `KSampler` /
   `KSamplerAdvanced` of the graph, else the first other known KSampler-like class, else
   the first node by number that has a sampler's inputs (custom samplers included). A
   graph without any gives no result; so does one whose sampler inputs cannot be read.

`--trace` shows the output node tried, the path back, and the sampler chosen with the
reason (first pass, nearest to the output, or fallback).

### Image Saver resources

PNGs saved by the Image Saver node (comfyui-image-saver) carry an A1111 `parameters`
text chunk next to the `prompt` graph. The graph stays the source of everything it holds
(prompts, sampler, seed, …; the chunk may spell them in Civitai's names). Only the
resources are taken from the chunk, so Civitai and hash lookups keep working:

| `meta` key (string) | `extra` key | Content |
|---|---|---|
| `Model hash` | `model_hash` | checkpoint hash |
| `Hashes` | `hashes` | dict: `model`, `LORA:<name>`, `embed:<name>` → hash |
| `Lora hashes` | `lora_hashes` | dict: LoRA name → hash |
| `Civitai resources` | `civitai_resources` | list of the resource objects |

Both Image Saver forms are read (older versions write only `Civitai resources`). The
chunk is parsed here rather than by IIB's regex, which splits unquoted JSON at its
commas. In the A1111 text the resources sit in the settings line around `Model`
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
With `--trace` it prints the path taken for every field instead: node id, class and
input name of each step, the branch a switch selected, the values read. That is what a
bug report needs. `trace_from_prompt(prompt)` returns the same as `(info, lines)`.

`extract_from_prompt(prompt)` returns a dict with `positive`, `negative`, `seed`,
`steps`, `cfg`, `sampler_name`, `scheduler`, `model_name` (as in the graph), `width`,
`height` and `loras` (list of `(name, weight)`). Unresolved values are `""`, `None` or
`[]`; a field whose walk fails is reported as unresolved without affecting the others.
It gives `None` when the graph has no sampler or the input is not a prompt dict.

`format_a1111(info)` turns the dict into an A1111-style `parameters` string. Its parts
are available too: `a1111_prompt_text(info)` (prompt and negative prompt, LoRAs
appended as `<lora:name:weight>` tags), `a1111_params(info)` (the settings as ordered
`(key, value)` strings; `Model` is the file name without folder and extension),
`lora_entries(info)` and `model_basename(name)`.
`image_saver_resources(parameters_text)` reads the [Image Saver
resources](#image-saver-resources) from an A1111 text; passed as second argument to
`format_a1111` and `a1111_params` they are added to the output.

`graphwalk.__version__` is the version of the core; the [changelog](CHANGELOG.md)
lists what changed. Import `graphwalk`, not `main`: `main.py` needs IIB.

## Supported nodes

What a node does is data in the rule tables near the top of
[graphwalk.py](graphwalk.py). The sections below explain the rules; the
[node lists](#node-lists) at the end are generated from the tables.

### Samplers

Any node counts as a sampler by its class (the KSampler variants, `SamplerCustom`,
`SamplerCustomAdvanced`) or, for other classes, by its inputs: those of a KSampler, or
those of `SamplerCustom` or `SamplerCustomAdvanced`.

A KSampler reads seed, steps, CFG, sampler and scheduler from its own widgets (or the
nodes wired to them, see [Sampler scalars](#sampler-scalars)). The seed is
`noise_seed` where the sampler has that input, else `seed`.

The Efficient samplers are recognised, but their prompts, size and LoRAs usually sit
on the `Efficient Loader` (string widgets for positive / negative, empty-latent size,
LoRA widgets), which is not read: with that loader, only the checkpoint name, seed,
steps, CFG, sampler and scheduler come out.

### Custom samplers

`SamplerCustom` and `SamplerCustomAdvanced` take most of their settings from nodes wired
to them: guider, noise, sampler and sigmas. Each of these edges is followed like any
other (`Reroute` and [switches](#switches) included) to the node it comes from; a node
that is not in the lists is not read, and its fields stay empty.

- **Guider** (`SamplerCustomAdvanced`): gives positive and negative prompt and CFG
  from the inputs the table names. `BasicGuider` has the positive prompt only (Flux's
  guidance is not a CFG scale and is not extracted, see
  [Conditioning](#conditioning)). `DualCFGGuider` has two CFG scales and a second
  conditioning with no counterpart in an A1111 line, so only `cond1` and `negative`
  are read. `LTXVDualCFGGuider`'s `video_cfg` is the CFG.
- **Noise**: `RandomNoise` gives the seed, `DisableNoise` none (`SamplerCustom` with
  `add_noise` off likewise).
- **Sampler**: `KSamplerSelect` gives its `sampler_name`; the other nodes stand for a
  sampler of their own (`euler_ancestral`, `lms`, …). The SDE samplers add `_gpu`
  unless their `noise_device` is `cpu`, as the nodes do; an unresolved `noise_device`
  leaves the name unresolved.
- **Sigmas**: `BasicScheduler` gives its `scheduler` widget and `steps`; for the other
  scheduler nodes the scheduler is named after the node (`karras`, `exponential` and
  `beta` as in ComfyUI's own list, the rest by the node's short name). `FlipSigmas` and
  `SetFirstSigma` are passed through. `SplitSigmas`, `SplitSigmasDenoise` and
  `ExtendIntermediateSigmas` are passed through too, but the number of steps is no
  longer the scheduler's, so `steps` stays empty. Without a known guider, the model
  is that of the scheduler node, if it has one.

The model, and so the checkpoint name and the LoRAs, comes from the guider's `model`
(through `ModelSampling*`, `CFGNorm`, LoRA loaders and the like, as for a KSampler); the
latent, and so the size, from `latent_image`.

### Prompt text

The prompt is reached by following the `positive` / `negative` conditioning edges (of the
sampler, or of its guider) to a text encoder, then following each of the encoder's text
inputs to a string. Encoders with several text inputs give all of them, in the order of
the [node lists](#node-lists); equal texts count once, different ones are joined with a
line break.

A text edge is followed recursively until a string is produced:

| Node | Behaviour |
|---|---|
| `Reroute` | transparent wire redirect |
| `ImpactWildcardProcessor` | reads `populated_text` (expanded wildcards at `mode='reproduce'`) |
| `StringFunction` / `StringFunction\|pysssss` | as the node's own code: `append` joins the non-empty `text_a`, `text_b`, `text_c` with `", "` when `tidy_tags` is `yes`, else with nothing; `replace` replaces `text_b` in `text_a` by `text_c` (`/regex/` supported); `tidy_tags` collapses blanks and repeated commas |
| `StringConcatenate` | `string_a` + `delimiter` + `string_b` as in core (delimiter default empty, empty parts kept) |
| `RegexReplace` | follows upstream text, then applies the substitution; pattern, replacement, count and flags may be wired from other nodes |
| Passthrough nodes (`TEXT_PASSTHROUGH`) | output one of their own text inputs unchanged |
| Switches | see [Switches](#switches) |
| Display nodes | follow the passthrough input; the cached widget is the fallback when the source cannot be resolved (see [Sampler scalars](#sampler-scalars)) |
| Primitive / `Note` / `TriggerWord Toggle` / … | any node storing a scalar in `value`, `text`, `string`, `populated_text` or `trigger_words` |

### Conditioning

- **Passthrough** (continue upstream): the nodes in the lists. `FluxGuidance` only sets
  a `guidance` value on the conditioning; it is not part of the result. An unknown
  node with a `conditioning` input is followed the same way (marked "generic" in the
  trace); one with several conditioning inputs is not.
- **Merge**: all inputs are joined.
- **Slot-aware passthrough**, one rule for every node that has both a `positive` and a
  `negative` input (`ControlNetApplyAdvanced`, `ControlNetApplySD3`,
  `InstructPix2PixConditioning`, `WanImageToVideo`, …): output slot 0 comes from
  `positive`, slot 1 from `negative`, any later slot is not conditioning. The deprecated
  single-input `ControlNetApply` follows its `conditioning` input.
- **`ConditioningZeroOut`** yields no text: an empty negative prompt stays empty
  instead of repeating the positive one.

### Switches

Followed on text, conditioning, model, latent and scalar edges. Only the switches in the
`SWITCHES` table of [graphwalk.py](graphwalk.py) are known; a node that merely has
"Switch" in its name is not guessed at (its inputs stay unresolved). The node lists say
how each one picks its input. Details:

- `ComfySwitchNode` / `ComfySoftSwitchNode`: `switch` is a literal or wired from a scalar
  node. A flag that cannot be resolved selects nothing.
- `ImpactSwitch`, `LatentSwitch`: `select` is a literal or wired. No such input, or no
  `select`, gives nothing, as the node itself outputs nothing then.
- The inversed switches have one input and several outputs: the one output slot that
  `select` / `index` names carries it, the others nothing.
- Easy-Use index switches count from 0; `easy textSwitch` takes 1 or 2.

### Sampler scalars

Seed, steps, CFG, sampler and scheduler are read from the sampler's own widgets, or
resolved along graph edges when wired from upstream nodes. The resolver carries the
*name* of the field being resolved so it can:

- follow [switches](#switches), `Reroute` and `Any to String (Image Saver)`, and read
  plain value nodes (`Primitive*`, `easy int` / `easy float`, …) from their `value` /
  `int` / `float` / `number` / `result` widget. A `seed` widget (`Seed (rgthree)`)
  answers only a request for the seed, so `steps` never comes from a seed; string
  fields (sampler, scheduler, file names) read only `value`;
- map the output slot of a multi-output node to the widget behind it (the Image Saver
  selectors, literals and `Input Parameters`, see the node lists);
- as the rule of last resort, match a widget named like the requested field on an
  aggregator / pipe node;
- resolve display nodes (`easy showAnything`, `ShowText`, `PreviewAny`, …) by following
  their input edge. Their cached widget is a leftover of an earlier run and goes stale
  when the source is edited, so it is read only when the source cannot be resolved.

### Model

- **Loaders** end the model chain. The file name is in `ckpt_name` or `unet_name`, a
  literal or wired from another node. A node of an unknown class at the end of the
  chain (no `model` input) is read through `ckpt_name` / `unet_name`; other widgets
  such as `model_name` (upscalers, detectors) do not end it.
- **Passthrough** (continue upstream): **any** node carrying a `model` edge is followed
  generically, with no whitelist. This covers LoRA loaders, model patchers
  (`CLIPSetLastLayer`, `ModelSamplingDiscrete`, `FreeU`, `RescaleCFG`,
  `DifferentialDiffusion`, …), CLIP-conditioning patchers (`CLIPNegPip`), IPAdapter
  variants and `Reroute`, plus any future node that merely forwards the model. Switch
  nodes on the model edge follow their selected branch.
- **Model merges** (`ModelMergeSimple`, `ModelMergeAdd`, `ModelMergeBlocks`, … with
  `model1` / `model2`): the model name is that of `model1`; LoRAs are collected from
  both inputs.
- `model_name` is the name as stored in the graph; `Model` in the parameters line is the
  file name without folder (`/` or `\`) and extension.

### LoRAs

The model chain is also walked to collect every applied LoRA, which is emitted as
A1111-style `<lora:name:weight>` tags appended to the positive prompt and handed to
IIB's indexer as the LoRA list, which turns it into searchable `lora` tags (tag search,
filters, LoRA statistics) the same way as for A1111 images. ComfyUI keeps LoRAs in
loader nodes rather than in the prompt text, so without this they are invisible.
LoRA-loader nodes understood:

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

### Image size

Width and height are read from the empty-latent node the sampler's latent starts at
(direct or Primitive-referenced scalars). The known `Empty…Latent…` classes are in the
node lists; any other node whose class name starts with `Empty`, contains `Latent` and
has `width` and `height` inputs counts the same way. For the choice of the [first
pass](#which-sampler-is-read) any `Empty…Latent…` node is a start, with or without a
size. A latent that does not start at an empty-latent node (img2img) gives no size.

When width / height are wired from a size-picker node without `width` / `height` widgets,
the size is read from the first `W×H` / `WxH` in one of its literal string widgets (e.g.
a preset label `Portrait 2:3 · 1152×1728 · 1.9 MP`, width first); without one, a widget
whose name contains `width` / `height` (e.g. `custom_width`) is used. This applies only
to the node at the end of the `width` / `height` edge of an empty-latent node, never to
other fields. Pickers that compute the size from aspect ratio and megapixels are not
covered.

`Reroute` and the [switch nodes](#switches) are followed automatically.

### Node lists

Generated from the rule tables by `tools/supported_nodes.py --write`; do not edit by
hand.

<!-- supported-nodes:begin -->

#### Samplers

- KSampler-like: `BNK_Unsampler` · `KSampler` · `KSampler (Efficient)` · `KSamplerAdvanced` · `KSamplerAdvanced (Efficient)` · `Tiled KSampler`
- With guider, sampler, sigmas and noise as nodes: `SamplerCustom` · `SamplerCustomAdvanced`
- Any other node counts as a sampler by its inputs, when it has all of
  - `latent_image` · `model` · `negative` · `positive` · `steps` and `seed` or `noise_seed` (as a KSampler)
  - `latent_image` · `model` · `negative` · `noise_seed` · `positive` · `sampler` · `sigmas` (as `SamplerCustom`)
  - `guider` · `latent_image` · `noise` · `sampler` · `sigmas` (as `SamplerCustomAdvanced`)

#### Output nodes

- Save nodes taking the pictures at `images`: `SaveImage` · `SaveImageAdvanced` · `SaveAnimatedWEBP` · `SaveAnimatedPNG` · `SaveWEBM` · `Image Saver` · `Image Saver Simple` · `Image Saver (From Pipe)` · `Save Image (LoraManager)` · `SaveImageKJ` · `SaveImageWithAlpha` · `easy imageSave` · `SaveImageNaming`
- Save nodes taking the pictures at `video`: `SaveVideo`
- Preview nodes: `ImageSender` · `PreviewAnimation` · `PreviewBridge` · `PreviewImage` · `SaveImageWebsocket`
- Inputs followed from a picture back to its latent: `images` · `image` · `pixels` · `video` · `latent` · `latent_image` · `samples` · `LATENT`

#### Text encoders

- `CLIPTextEncode`: `text`
- `CLIPTextEncodeSDXL`: `text_g` · `text_l`
- `CLIPTextEncodeSDXLRefiner`: `text`
- `smZ CLIPTextEncode`: `text_g` · `text_l` · `text`
- `BNK_CLIPTextEncodeAdvanced`: `text_g` · `text_l` · `text`
- `CLIPTextEncodeFlux`: `clip_l` · `t5xxl`
- `CLIPTextEncodeSD3`: `clip_l` · `clip_g` · `t5xxl`
- `CLIPTextEncodeHiDream`: `clip_l` · `clip_g` · `t5xxl` · `llama`
- `CLIPTextEncodeHunyuanDiT`: `bert` · `mt5xl`
- `CLIPTextEncodeKandinsky5`: `clip_l` · `qwen25_7b`
- `CLIPTextEncodeLumina2`: `user_prompt`
- `CLIPTextEncodePixArtAlpha`: `text`
- `TextEncodeHunyuanVideo_ImageToVideo`: `prompt`
- `TextEncodeQwenImageEdit`: `prompt`
- `TextEncodeQwenImageEditPlus`: `prompt`

#### Conditioning

- Passthrough: `ConditioningSetArea` · `ConditioningSetAreaPercentage` · `ConditioningSetAreaStrength` · `ConditioningSetMask` · `ConditioningSetTimestepRange` · `FluxGuidance` · `FluxKontextMultiReferenceLatentMethod` · `ReferenceLatent` · `unCLIPConditioning`
- Merge, all inputs (`conditioning_1` · `conditioning_2` · `conditioning_to` · `conditioning_from`): `ConditioningAverage` · `ConditioningCombine` · `ConditioningConcat`
- No text, whatever feeds them: `ConditioningZeroOut`
- Any node with both a `positive` and a `negative` input: output slot 0 comes from `positive`, slot 1 from `negative`

#### Custom sampler inputs

- Guiders:
  - `CFGGuider`: positive `positive`, negative `negative`, cfg `cfg`
  - `BasicGuider`: positive `conditioning`, no negative, no cfg
  - `DualCFGGuider`: positive `cond1`, negative `negative`, no cfg
  - `DualModelGuider`: positive `positive`, negative `negative`, cfg `cfg`
  - `PerpNegGuider`: positive `positive`, negative `negative`, cfg `cfg`
  - `LTXVDualCFGGuider`: positive `positive`, negative `negative`, cfg `video_cfg`
  - `ScheduledCFGGuidance`: positive `positive`, negative `negative`, cfg `cfg`
- Noise:
  - `RandomNoise`: seed `noise_seed`
  - `DisableNoise`: no seed
- Samplers:
  - `KSamplerSelect`: `sampler_name` widget
  - `SamplerEulerAncestral`: `euler_ancestral`
  - `SamplerEulerAncestralCFGPP`: `euler_ancestral_cfg_pp`
  - `SamplerDPMPP_2S_Ancestral`: `dpmpp_2s_ancestral`
  - `SamplerLMS`: `lms`
  - `SamplerDPMAdaptative`: `dpm_adaptive`
  - `SamplerER_SDE`: `er_sde`
  - `SamplerDPMPP_3M_SDE`: `dpmpp_3m_sde`, with `_gpu` unless `noise_device` is `cpu`
  - `SamplerDPMPP_2M_SDE`: `dpmpp_2m_sde`, with `_gpu` unless `noise_device` is `cpu`
  - `SamplerDPMPP_SDE`: `dpmpp_sde`, with `_gpu` unless `noise_device` is `cpu`
- Sigmas, as schedulers (all with `steps`):
  - `BasicScheduler`: `scheduler` widget
  - `KarrasScheduler`: `karras`
  - `ExponentialScheduler`: `exponential`
  - `PolyexponentialScheduler`: `polyexponential`
  - `LaplaceScheduler`: `laplace`
  - `BetaSamplingScheduler`: `beta`
  - `VPScheduler`: `vp`
  - `SDTurboScheduler`: `sd_turbo`
  - `AlignYourStepsScheduler`: `align_your_steps`
  - `GITSScheduler`: `gits`
  - `OptimalStepsScheduler`: `optimal_steps`
  - `Flux2Scheduler`: `flux2`
  - `LTXVScheduler`: `ltxv`
  - `Ideogram4Scheduler`: `ideogram4`
- Sigmas passed through: `FlipSigmas` · `SetFirstSigma`
- Sigmas passed through, `steps` left empty: `ExtendIntermediateSigmas` · `SplitSigmas` · `SplitSigmasDenoise`

#### Switches

- `ComfySwitchNode`: `on_true` / `on_false` by `switch`
- `ComfySoftSwitchNode`: `on_true` / `on_false` by `switch`; the connected one when only one is connected
- `Any Switch (rgthree)`: the first connected `any_<n>` input, by number
- `ImpactSwitch`: `input<n>` for the number `n` in `select`
- `LatentSwitch`: `input<n>` or `latent<n>` for the number `n` in `select`
- `ImpactInversedSwitch`: one input `input`, passed on only by the output slot numbered `select` - 1
- `easy anythingIndexSwitch`: `value<n>` for the number `n` in `index`
- `easy textIndexSwitch`: `text<n>` for the number `n` in `index`
- `easy conditioningIndexSwitch`: `cond<n>` for the number `n` in `index`
- `easy textSwitch`: `text<n>` for the number `n` in `input`
- `easy anythingInversedSwitch`: one input `in`, passed on only by the output slot numbered `index`

#### Text and values

- `LLMPromptEnhancer`: text passes through from `prompt`
- `Any to String (Image Saver)`: text and values pass through from `value`
- Display nodes, followed through the first connected of `anything` · `source` · `input` · `value` · `text`: `Display Any (rgthree)` · `Display Float (rgthree)` · `Display Int (rgthree)` · `DisplayAny` · `easy showAnything` · `PreviewAny` · `ShowText` · `ShowText|pysssss`
- Widgets of plain value nodes: text `value` · `text` · `string` · `populated_text` · `trigger_words`; seeds `value` · `int` · `float` · `seed` · `number` · `result`; names `value`; other numbers `value` · `int` · `float` · `number` · `result`
- Cached widgets of display nodes, read when the source cannot be resolved: text `text` · `value` · `string`; values `text` · `value` · `int` · `float` · `string`
- Multi-output nodes, output slot by slot:
  - `Input Parameters (Image Saver)`: `seed`, `steps`, `cfg`, `sampler`, `scheduler`, `denoise`
  - `Sampler Selector (Image Saver)`: `sampler_name`, `sampler_name`
  - `Scheduler Selector (Image Saver)`: `scheduler`, `scheduler`
  - `Scheduler Selector (inspire) (Image Saver)`: `scheduler`, `scheduler`
  - `Scheduler Selector (Eff.) (Image Saver)`: `scheduler`, `scheduler`
  - `Checkpoint Loader with Name (Image Saver)`: -, -, -, `ckpt_name`
  - `UNet loader with Name (Image Saver)`: -, `unet_name`
  - `String Literal (Image Saver)`: `string`
  - `Int Literal (Image Saver)`: `int`
  - `Float Literal (Image Saver)`: `float`
  - `Cfg Literal (Image Saver)`: `cfg`
  - `Width/Height Literal (Image Saver)`: `size`

#### Models

- Loaders with the file name in `ckpt_name`: `CheckpointLoaderSimple` · `CheckpointLoader` · `unCLIPCheckpointLoader` · `ImageOnlyCheckpointLoader` · `Checkpoint Loader` · `CheckpointLoaderNF4` · `Checkpoint Loader (LoraManager)` · `Random Checkpoint Loader (LoraManager)` · `CheckpointLoaderSimpleMultiGPU` · `Checkpoint Loader with Name (Image Saver)`
- Loaders with the file name in `unet_name`: `UNETLoader` · `UNet loader with Name (Image Saver)` · `UnetLoaderGGUF` · `UnetLoaderGGUFAdvanced` · `Unet Loader (LoraManager)` · `Random Unet Loader (LoraManager)` · `UNETLoaderMultiGPU` · `UnetLoaderGGUFMultiGPU`
- A node of any other class at the end of the model chain: `ckpt_name` · `unet_name`

#### Empty latents

- `EmptyARVideoLatent` · `EmptyChromaRadianceLatentImage` · `EmptyCosmosLatentVideo` · `EmptyFlux2LatentImage` · `EmptyHiDreamO1LatentImage` · `EmptyHunyuanImageLatent` · `EmptyHunyuanLatentVideo` · `EmptyHunyuanVideo15Latent` · `EmptyLatentImage` · `EmptyLTXVLatentVideo` · `EmptyMiniMaxH3LatentAV` · `EmptyMochiLatentVideo` · `EmptyQwenImageLayeredLatentImage` · `EmptySD3LatentImage`

<!-- supported-nodes:end -->

## Adding a node

The walker is generic. Most gaps are fixed by adding a class name to the right table
near the top of [graphwalk.py](graphwalk.py):

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

Then run `python3 tools/supported_nodes.py --write` to update the [node lists](#node-lists);
the tests fail while the README is out of date. Nodes that transform text (concatenate,
regex, wildcards, …) need a dedicated branch in `Walker._text_of`. PRs welcome.

## Tests

Core tests, standard library only:

```bash
python3 -m unittest discover tests
```

CI runs this on Python 3.10 to 3.13 (`.github/workflows/tests.yml`). `tests/test_main.py`
tests the IIB adapter; it needs IIB, so it is skipped there and in any other environment
without IIB. Run it inside an IIB checkout or container, e.g. with the Docker image of IIB:

```bash
docker run --rm -v "$PWD":/app/plugins/comfyui_graphwalk:ro -w /app <iib-image> \
    python -m unittest discover -s /app/plugins/comfyui_graphwalk/tests
```

All graphs and prompts in the tests are synthetic: `test_graphwalk.py` tests the
behaviour of the whole extraction, `test_rules.py` one rule table and heuristic at a
time, `test_sampler.py` the choice of the sampler (hires fix, refiner, img2img,
several output nodes, fallback), `test_custom_sampler.py` `SamplerCustom(Advanced)`
with every guider, noise, sampler and scheduler node, `test_models.py` the encoders
and empty latents of newer models, `test_trace.py` the trace mode and the command
line, `test_supported_nodes.py` that the [node lists](#node-lists) match the rule
tables.

To check that a change leaves the output for your own images unchanged, compare two
versions of the core with `tools/compare_versions.py` (Pillow required). It prints only
counts, file paths and the names of differing fields, never prompt text:

```bash
git show main:graphwalk.py > /tmp/graphwalk_old.py
python3 tools/compare_versions.py /tmp/graphwalk_old.py graphwalk.py /path/to/images
```

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Mikolas Bingemer.
