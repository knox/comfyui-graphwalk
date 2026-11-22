# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are in
`graphwalk.__version__`.

## [2.0.0] - 2026-11-22

The public interface of `graphwalk.py` (`extract_from_prompt`, `format_a1111` and the
other names in `__all__`) is unchanged, its result dict only gains `clip_skip`; the
output and the internals change. The internal functions of 1.x (`resolve_text`,
`collect_loras`, `get_scalar`, …) are gone.

### Changed

- **Breaking: the scheduler is its own key.** `Sampler: euler simple` is now
  `Sampler: euler, Schedule type: simple`, as A1111 writes it, and the metadata has a
  `Schedule type` entry. Images indexed with 1.x keep the old text until they are
  parsed again: re-parse them, see [Refresh already-indexed
  images](README.md#refresh-already-indexed-images-one-time).
- The sampler read is the one that made the saved picture, found from the output node
  back through decode and upscale to the first pass, instead of the first `KSampler` of
  the graph. Graphs with hires fix, refiner or detailer samplers can give other values
  than before.
- One walker with rule tables replaces the six resolvers of 1.x. Where a rule was
  guessed before, the value now stays empty (see Fixed).
- Prompt tags come from IIB's `parse_prompt` directly instead of a stand-in parameters
  line (IIB without `parse_prompt` keeps the old way).
- The README lists the supported nodes from the rule tables.

### Added

- `SamplerCustom` and `SamplerCustomAdvanced`, with guider, noise, sampler and sigmas
  nodes.
- Text encoders of newer models (Flux, SD3, HiDream, HunyuanDiT, Kandinsky 5, Lumina 2,
  PixArt, Qwen image edit, Hunyuan video) and empty latents of newer models; any
  `Empty…Latent…` node with `width` and `height` gives the size.
- Resources of Image Saver PNGs: `Model hash`, `Hashes`, `Lora hashes` and
  `Civitai resources`, in the metadata and in `extra`; `image_saver_resources()` reads
  them from an A1111 text.
- Clip skip: `clip_skip` in the result of `extract_from_prompt`, the negated layer of the
  first `CLIPSetLastLayer` on the CLIP path of the positive prompt's text encoder
  (through LoRA loaders, `Reroute`, switches and any node with a `clip` input), `None`
  without one; `Clip skip: N` in the parameters line and the metadata for N > 1, after
  `Model`, as A1111 writes it. It is not a tag, and the `Clip skip` of an Image Saver
  `parameters` chunk is not used.
- Trace mode: `python3 graphwalk.py <image.png | prompt.json> [--trace]` and
  `trace_from_prompt()`.
- `graphwalk.__version__`, `tools/supported_nodes.py`, more tests and CI on Python 3.10
  to 3.13.

### Fixed

- Display nodes (`easy showAnything`, `ShowText`, …) gave the value cached from an
  earlier run, which is stale once the source was edited. The input edge is followed
  first; the cached widget is only the fallback.
- Heuristics that returned wrong values now leave them empty:
  - switches are the ones in the `SWITCHES` table, not every class with "Switch" in its
    name; Impact switches without a usable `select` and unresolved `switch` flags select
    nothing instead of the first input or `on_false`;
  - a `seed` widget answers only a seed request, string fields read only `value`;
  - the `W×H` in a size label is read only at the end of an empty-latent `width` /
    `height` edge;
  - the model name comes from the model loaders (or `ckpt_name` / `unet_name` of an
    unknown source), not from any `model_name` widget;
  - an unknown node with several conditioning inputs is no longer followed half-way.
- A sampler whose inputs cannot be read gives no result (and so IIB's parser) instead of
  raising.

## [1.x]

Earlier: see git history.
