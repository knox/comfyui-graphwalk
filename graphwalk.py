#!/usr/bin/env python3
"""
ComfyUI Graph-Walk core: generation parameters from a ComfyUI ``prompt`` graph.

Finds the sampler a picture was made by (from the output node back through decode
and upscale to the first pass, see ``_find_output_sampler``), walks the ComfyUI API
``prompt`` graph backwards from it, resolving the real values through Reroute /
LoRA-loader / conditioning / switch chains, and formats them as an A1111-style
``parameters`` string. KSampler-like samplers and SamplerCustom(Advanced) are read.

Standard library only, no IIB or PIL: the caller reads the ``prompt`` chunk from
the image and passes the parsed dict. The IIB plugin (``main.py``) is a thin
adapter around this module. Public interface:

- ``extract_from_prompt(prompt: dict) -> dict | None``
- ``format_a1111(info: dict) -> str``

Command line (debugging, bug reports)::

    python3 graphwalk.py <image.png | prompt.json> [--trace]

prints the A1111 result, or with ``--trace`` the path taken for every field.

How it is built: one walker (``Walker``) follows an edge ``[node_id, slot]``
backwards through the nodes that merely pass a signal on, and stops at the first
node that really is the source. What "passes a signal on" means is decided per
signal type (CONDITIONING, MODEL, CLIP, LATENT, STRING, scalar, IMAGE and the custom
sampler's GUIDER, SAMPLER, SIGMAS, NOISE) by ``Walker._next_*`` and the rule tables
below. Consumers (``cond_text``, ``text``, ``scalar``, ``model_name``, ``loras``,
``clip_skip``, ``latent_size``) read the source node they arrive at. The walker does not know the
sampler: any edge can be resolved.

Principle: better empty than wrong. Where a rule is not known, the value stays
unresolved instead of being guessed.

See README.md for behaviour and supported nodes.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Mikolas Bingemer
"""

import json
import re
import struct
import sys
import zlib
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import NamedTuple

__version__ = "2.0.0"

__all__ = [
    "extract_from_prompt",
    "format_a1111",
    "a1111_prompt_text",
    "a1111_params",
    "lora_entries",
    "model_basename",
    "trace_from_prompt",
    "image_saver_resources",
]

# ---------------------------------------------------------------------------
# Signal types
# ---------------------------------------------------------------------------

COND = "CONDITIONING"
MODEL = "MODEL"
CLIP = "CLIP"  # the clip input of the first text encoder, back to the CLIPSetLastLayer that set its layer
LATENT = "LATENT"
TEXT = "STRING"
SCALAR = "SCALAR"  # seed, steps, cfg, names, flags, sizes: a literal widget value
LORA_STACK = "LORA_STACK"  # LoraManager's stack of LoRAs, passed from Stacker to Stacker
IMAGE = "IMAGE"  # a picture on its way back from an output node through the decode to the first latent
# The inputs of the custom samplers (SamplerCustom, SamplerCustomAdvanced), each a node of its own.
GUIDER = "GUIDER"
SAMPLER = "SAMPLER"
SIGMAS = "SIGMAS"
NOISE = "NOISE"

# ---------------------------------------------------------------------------
# Rule tables: which node classes do what
# ---------------------------------------------------------------------------

# KSampler-like samplers: ``model``, ``positive``, ``negative``, ``latent_image``, a seed and
# ``steps`` as widgets. Known classes; any other node with that signature counts as well
# (see ``_sampler_kind``).
SAMPLER_CLASSES = {
    "KSampler",
    "KSamplerAdvanced",
    # efficiency-nodes
    "KSampler (Efficient)",
    "KSamplerAdvanced (Efficient)",
    "Tiled KSampler",
    "BNK_Unsampler",
}
# Samplers whose guider, sampler, sigmas and noise are nodes of their own.
CUSTOM_SAMPLERS = {"SamplerCustom": "custom", "SamplerCustomAdvanced": "advanced"}
# The signatures (all of these inputs, plus a seed and ``steps`` for the KSampler kind).
SAMPLER_SIGNATURE = frozenset({"model", "positive", "negative", "latent_image", "steps"})
CUSTOM_SIGNATURE = frozenset({"model", "positive", "negative", "sampler", "sigmas", "latent_image", "noise_seed"})
ADVANCED_SIGNATURE = frozenset({"noise", "guider", "sampler", "sigmas", "latent_image"})

# --- Output nodes (where the search for the sampler starts) ----------------

# Nodes that save pictures: class -> the input that takes them. Any other node with an
# ``images`` input that nothing else uses is an output node as well (a video combiner, say).
SAVE_CLASSES = {
    # core
    "SaveImage": "images",
    "SaveImageAdvanced": "images",
    "SaveAnimatedWEBP": "images",
    "SaveAnimatedPNG": "images",
    "SaveWEBM": "images",
    "SaveVideo": "video",
    # ComfyUI-Image-Saver
    "Image Saver": "images",
    "Image Saver Simple": "images",
    "Image Saver (From Pipe)": "images",
    # ComfyUI-Lora-Manager, comfyui-kjnodes, ComfyUI-Easy-Use, NamingSave
    "Save Image (LoraManager)": "images",
    "SaveImageKJ": "images",
    "SaveImageWithAlpha": "images",
    "easy imageSave": "images",
    "SaveImageNaming": "images",
}
# Nodes that only show pictures. IIB indexes saved images, so these are used only when
# no save node leads to a sampler.
PREVIEW_CLASSES = {
    "PreviewImage",
    "PreviewAnimation",  # comfyui-kjnodes
    "PreviewBridge",  # Impact Pack
    "ImageSender",  # Impact Pack
    "SaveImageWebsocket",  # saves nothing
}
# Inputs a node on the picture's way back takes its picture (or latent) from; the first
# one that is connected is followed. VAEDecode(Tiled) takes ``samples``, VAEEncode
# ``pixels``, upscalers ``image``, CreateVideo ``images``, GetVideoComponents ``video``.
IMAGE_INPUTS = ("images", "image", "pixels", "video")

# --- CONDITIONING ----------------------------------------------------------

# Nodes that pass the ``conditioning`` input on unchanged (no text of their own).
# Unknown nodes with a ``conditioning`` input are followed as well (marked
# "generic" in the trace), this list is the known part.
COND_PASSTHROUGH = {
    "ConditioningSetTimestepRange",
    "ConditioningSetArea",
    "ConditioningSetAreaPercentage",
    "ConditioningSetAreaStrength",
    "ConditioningSetMask",
    "unCLIPConditioning",
    # Flux: FluxGuidance only stores a ``guidance`` value, which is not extracted
    "FluxGuidance",
    "FluxKontextMultiReferenceLatentMethod",
    "ReferenceLatent",
}

# Text encoders (the end of the conditioning chain): class -> the inputs holding the
# prompt, in the order they are read. Equal texts count once, different ones are
# joined with a line break (SDXL's global and local prompt, Flux's CLIP-L and T5 prompt).
_LEGACY_TEXT_KEYS = ("text_g", "text_l", "text")  # smZ, BNK: read as before
TEXT_ENCODERS = {
    "CLIPTextEncode": ("text",),
    "CLIPTextEncodeSDXL": ("text_g", "text_l"),
    "CLIPTextEncodeSDXLRefiner": ("text",),
    "smZ CLIPTextEncode": _LEGACY_TEXT_KEYS,
    "BNK_CLIPTextEncodeAdvanced": _LEGACY_TEXT_KEYS,
    "CLIPTextEncodeFlux": ("clip_l", "t5xxl"),
    "CLIPTextEncodeSD3": ("clip_l", "clip_g", "t5xxl"),
    "CLIPTextEncodeHiDream": ("clip_l", "clip_g", "t5xxl", "llama"),
    "CLIPTextEncodeHunyuanDiT": ("bert", "mt5xl"),
    "CLIPTextEncodeKandinsky5": ("clip_l", "qwen25_7b"),
    "CLIPTextEncodeLumina2": ("user_prompt",),
    "CLIPTextEncodePixArtAlpha": ("text",),
    "TextEncodeHunyuanVideo_ImageToVideo": ("prompt",),
    "TextEncodeQwenImageEdit": ("prompt",),
    "TextEncodeQwenImageEditPlus": ("prompt",),
}

# Nodes that merge several conditioning inputs into one: the texts of all of them.
COND_COMBINE = {
    "ConditioningCombine",
    "ConditioningConcat",
    "ConditioningAverage",
}
COND_COMBINE_KEYS = ("conditioning_1", "conditioning_2", "conditioning_to", "conditioning_from")

# Conditioning sources without text, whatever feeds them: ConditioningZeroOut
# blanks the conditioning.
COND_BLANK = {"ConditioningZeroOut"}

# Generic rule, no class list: a node with both a ``positive`` and a ``negative``
# input (ControlNetApplyAdvanced / SD3, InstructPix2PixConditioning,
# WanImageToVideo, SD_4XUpscale_Conditioning, …) passes output slot 0 from
# ``positive`` and output slot 1 from ``negative``; later slots are not conditioning.
POS_NEG_SLOTS = ("positive", "negative")

# --- MODEL -----------------------------------------------------------------

# Loader classes that end the model chain: class -> widgets holding the file name.
MODEL_LOADERS = {
    "CheckpointLoaderSimple": ("ckpt_name",),
    "CheckpointLoader": ("ckpt_name",),
    "unCLIPCheckpointLoader": ("ckpt_name",),
    "ImageOnlyCheckpointLoader": ("ckpt_name",),
    "Checkpoint Loader": ("ckpt_name",),
    "CheckpointLoaderNF4": ("ckpt_name",),
    "Checkpoint Loader (LoraManager)": ("ckpt_name",),
    "Random Checkpoint Loader (LoraManager)": ("ckpt_name",),
    "CheckpointLoaderSimpleMultiGPU": ("ckpt_name",),
    "Checkpoint Loader with Name (Image Saver)": ("ckpt_name",),
    "UNETLoader": ("unet_name",),
    "UNet loader with Name (Image Saver)": ("unet_name",),
    "UnetLoaderGGUF": ("unet_name",),
    "UnetLoaderGGUFAdvanced": ("unet_name",),
    "Unet Loader (LoraManager)": ("unet_name",),
    "Random Unet Loader (LoraManager)": ("unet_name",),
    "UNETLoaderMultiGPU": ("unet_name",),
    "UnetLoaderGGUFMultiGPU": ("unet_name",),
}
# A node of an unknown class at the end of the model chain (no ``model`` input at
# all, so it must create the model) is read through these widgets. Not
# ``model_name``: upscale models, detectors and many others use it.
MODEL_SOURCE_WIDGETS = ("ckpt_name", "unet_name")

# --- CLIP ------------------------------------------------------------------

# Nodes that set the last CLIP layer (clip skip): class -> the widget holding it. The first
# one on the way back from a text encoder counts, the walk ends there. The widget is -clip skip.
CLIP_SOURCES = {
    "CLIPSetLastLayer": "stop_at_clip_layer",
}
# Nodes that pass the ``clip`` input on to their CLIP output (the LoRA loaders). Unknown
# nodes with a ``clip`` input are followed as well (marked "generic" in the trace), this
# list is the known part. A loader that creates the CLIP model ends the walk without a value.
CLIP_PASSTHROUGH = {
    "LoraLoader",
    "Power Lora Loader (rgthree)",
    "Lora Loader Stack (rgthree)",
    "Lora Loader (LoraManager)",
    "LoRA Text Loader (LoraManager)",
}

# --- LATENT ----------------------------------------------------------------

# Empty-latent nodes: where a generation starts, and the source of width/height. Any
# other node named ``Empty…Latent…`` with ``width`` and ``height`` inputs counts as a
# size source as well (see ``_is_size_source``); these are the known ones.
LATENT_SIZE_CLASSES = {
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyHunyuanLatentVideo",
    "EmptyHunyuanVideo15Latent",
    "EmptyHunyuanImageLatent",
    "EmptyFlux2LatentImage",
    "EmptyQwenImageLayeredLatentImage",
    "EmptyChromaRadianceLatentImage",
    "EmptyHiDreamO1LatentImage",
    "EmptyLTXVLatentVideo",
    "EmptyMochiLatentVideo",
    "EmptyCosmosLatentVideo",
    "EmptyARVideoLatent",
    "EmptyMiniMaxH3LatentAV",
}
# Inputs a latent-to-latent node (upscale, VAE round-trips, samplers, …) takes its
# latent from; the first one that is connected is followed.
LATENT_INPUTS = ("latent", "latent_image", "samples", "LATENT")

# A "W×H" / "WxH" size inside a label, e.g. "Portrait 2:3 · 1152×1728 · 1.9 MP".
SIZE_IN_LABEL = re.compile(r"(\d+)\s*[×x]\s*(\d+)")

# --- Inputs of the custom samplers ---------------------------------------------


class Guider(NamedTuple):
    """The inputs of a guider node that hold the positive and negative conditioning and the CFG."""

    positive: str | None
    negative: str | None = None
    cfg: str | None = None


# Guiders (SamplerCustomAdvanced's ``guider``); the model is always the ``model`` input.
# Anything that is not here is not read. DualCFGGuider has two CFG scales and a second
# conditioning with no counterpart in an A1111 line: only ``cond1`` and ``negative`` are read.
GUIDERS = {
    "CFGGuider": Guider("positive", "negative", "cfg"),
    "BasicGuider": Guider("conditioning"),  # no negative prompt, no CFG
    "DualCFGGuider": Guider("cond1", "negative"),
    "DualModelGuider": Guider("positive", "negative", "cfg"),
    "PerpNegGuider": Guider("positive", "negative", "cfg"),
    "LTXVDualCFGGuider": Guider("positive", "negative", "video_cfg"),
    "ScheduledCFGGuidance": Guider("positive", "negative", "cfg"),  # comfyui-kjnodes
}

# Noise nodes (``noise``): class -> the widget holding the seed (None: no noise, no seed).
NOISE_SEEDS = {
    "RandomNoise": "noise_seed",
    "DisableNoise": None,
}

# Sampler nodes (``sampler``): class -> the sampler name, None where the node has a
# ``sampler_name`` widget. The SDE samplers add ``_gpu`` unless their ``noise_device`` is
# ``cpu``, as the nodes do themselves.
SAMPLER_NODES = {
    "KSamplerSelect": None,
    "SamplerEulerAncestral": "euler_ancestral",
    "SamplerEulerAncestralCFGPP": "euler_ancestral_cfg_pp",
    "SamplerDPMPP_2S_Ancestral": "dpmpp_2s_ancestral",
    "SamplerLMS": "lms",
    "SamplerDPMAdaptative": "dpm_adaptive",
    "SamplerER_SDE": "er_sde",
    "SamplerDPMPP_3M_SDE": "dpmpp_3m_sde",
    "SamplerDPMPP_2M_SDE": "dpmpp_2m_sde",
    "SamplerDPMPP_SDE": "dpmpp_sde",
}
SAMPLER_NOISE_DEVICE = {"SamplerDPMPP_3M_SDE", "SamplerDPMPP_2M_SDE", "SamplerDPMPP_SDE"}

# Scheduler nodes (``sigmas``): class -> the name shown as the scheduler, None where the
# node has a ``scheduler`` widget. The names are those of ComfyUI's own schedulers where
# there is one (``karras``, ``exponential``, ``beta``; A1111 has the same), else the
# node's short name. All of them have a ``steps`` input.
SCHEDULERS = {
    "BasicScheduler": None,
    "KarrasScheduler": "karras",
    "ExponentialScheduler": "exponential",
    "PolyexponentialScheduler": "polyexponential",
    "LaplaceScheduler": "laplace",
    "BetaSamplingScheduler": "beta",
    "VPScheduler": "vp",
    "SDTurboScheduler": "sd_turbo",
    "AlignYourStepsScheduler": "align_your_steps",
    "GITSScheduler": "gits",
    "OptimalStepsScheduler": "optimal_steps",
    "Flux2Scheduler": "flux2",
    "LTXVScheduler": "ltxv",
    "Ideogram4Scheduler": "ideogram4",
}
# Nodes that take sigmas and give sigmas: the scheduler behind them is the source.
SIGMAS_PASSTHROUGH = {
    "FlipSigmas",
    "SetFirstSigma",
    "SplitSigmas",
    "SplitSigmasDenoise",
    "ExtendIntermediateSigmas",
}
# ... but these change the number of steps: the steps of the scheduler would be wrong.
SIGMAS_RESIZERS = {"SplitSigmas", "SplitSigmasDenoise", "ExtendIntermediateSigmas"}

# --- STRING / scalar -------------------------------------------------------

# Display / debug nodes that echo an upstream value into a cached widget and pass
# it through unchanged. The edge is followed first: what the source holds is what
# was executed. The cached widget is a leftover of an earlier run (it goes stale
# when the source is edited) and is read only when the source cannot be resolved.
SHOW_PASSTHROUGH = {
    "easy showAnything",
    "ShowText|pysssss",
    "ShowText",
    "Display Any (rgthree)",
    "DisplayAny",
    "PreviewAny",
    "Display Int (rgthree)",
    "Display Float (rgthree)",
}
SHOW_CACHE_KEYS = {
    TEXT: ("text", "value", "string"),
    SCALAR: ("text", "value", "int", "float", "string"),
}
SHOW_FOLLOW_KEYS = ("anything", "source", "input", "value", "text")  # the first one that is an edge

# Nodes that output one of their own text inputs unchanged, mapped to that input's
# name. For inputs not covered by the generic text keys ("value", "text", …) whose
# name can't be added there because it means something else on other nodes
# (TextGenerate's 'prompt' is the LLM's input, not its output).
TEXT_PASSTHROUGH = {
    "LLMPromptEnhancer": "prompt",
}
TEXT_KEYS = ("value", "text", "string", "populated_text", "trigger_words")

# Nodes with several outputs whose widgets are the values: output slot -> widget
# (None: the slot carries something else). The widget may be wired from another
# node. Beats the generic rule "a widget named like the requested field".
OUTPUT_WIDGETS = {
    # ComfyUI-Image-Saver (RETURN_NAMES in nodes_selectors.py / nodes_loaders.py /
    # nodes_literals.py)
    "Input Parameters (Image Saver)": ("seed", "steps", "cfg", "sampler", "scheduler", "denoise"),
    "Sampler Selector (Image Saver)": ("sampler_name", "sampler_name"),
    "Scheduler Selector (Image Saver)": ("scheduler", "scheduler"),
    "Scheduler Selector (inspire) (Image Saver)": ("scheduler", "scheduler"),
    "Scheduler Selector (Eff.) (Image Saver)": ("scheduler", "scheduler"),
    "Checkpoint Loader with Name (Image Saver)": (None, None, None, "ckpt_name"),
    "UNet loader with Name (Image Saver)": (None, "unet_name"),
    "String Literal (Image Saver)": ("string",),
    "Int Literal (Image Saver)": ("int",),
    "Float Literal (Image Saver)": ("float",),
    "Cfg Literal (Image Saver)": ("cfg",),
    "Width/Height Literal (Image Saver)": ("size",),
}

# Nodes that output their input unchanged, for text and scalars: class -> input.
VALUE_PASSTHROUGH = {
    "Any to String (Image Saver)": "value",
}

# Keys under which a plain value node (Primitive, Seed, easy int/float, …) stores
# its value, by the kind of field being resolved. A ``seed`` widget answers only a
# request for a seed; a string field (sampler, scheduler, file names) takes only
# ``value``, not an ``int``.
VALUE_KEYS = ("value", "int", "float", "number", "result")
SEED_FIELDS = ("seed", "noise_seed")
SEED_KEYS = ("value", "int", "float", "seed", "number", "result")
STRING_FIELDS = ("sampler_name", "scheduler", "lora_name", "ckpt_name", "unet_name")
STRING_KEYS = ("value",)


def _value_keys(field: str):
    if field in SEED_FIELDS:
        return SEED_KEYS
    if field in STRING_FIELDS or re.fullmatch(r"lora_\d+", field):
        return STRING_KEYS
    return VALUE_KEYS


# --- Switches --------------------------------------------------------------


class Switch(NamedTuple):
    """
    How a known switch node picks its input. ``kind``:

    - ``bool``: ``on_true`` / ``on_false`` by the ``switch`` flag; ``soft``: the
      same, but the one connected input wins when only one is connected.
    - ``first``: the first connected input ``any_N`` (by N).
    - ``index``: ``<prefix><n>`` for the number ``n`` read from the ``select``
      input; a missing input gives nothing.
    - ``inversed``: one ``input``, several outputs; output slot ``select - base``
      carries it, the others nothing.
    """

    kind: str
    select: str = ""
    prefixes: tuple = ()
    input: str = ""
    base: int = 0


# Only these are switches. A node that merely has "Switch" in its name is not
# guessed at: an unknown switch is an ordinary unknown node.
SWITCHES = {
    # comfy-core If/Else
    "ComfySwitchNode": Switch("bool"),
    "ComfySoftSwitchNode": Switch("soft"),
    # rgthree: outputs the first non-empty of its any_01, any_02, … inputs
    "Any Switch (rgthree)": Switch("first"),
    # ComfyUI-Impact-Pack (input{select}; no such input = no output). LatentSwitch
    # is the same node there; older versions named the inputs latent{n}.
    "ImpactSwitch": Switch("index", select="select", prefixes=("input",)),
    "LatentSwitch": Switch("index", select="select", prefixes=("input", "latent")),
    "ImpactInversedSwitch": Switch("inversed", select="select", input="input", base=1),
    # ComfyUI-Easy-Use (0-based index; textSwitch is 1-based, input 1 or 2)
    "easy anythingIndexSwitch": Switch("index", select="index", prefixes=("value",)),
    "easy textIndexSwitch": Switch("index", select="index", prefixes=("text",)),
    "easy conditioningIndexSwitch": Switch("index", select="index", prefixes=("cond",)),
    "easy textSwitch": Switch("index", select="input", prefixes=("text",)),
    "easy anythingInversedSwitch": Switch("inversed", select="index", input="in", base=0),
}

# --- LoRA ------------------------------------------------------------------

# Matches <lora:name:weight> and <lora:name:model_weight:clip_weight> (same
# pattern as LoraManager's parse_lora_syntax); only the model weight is kept.
_LORA_TAG_RE = re.compile(r"<lora:([^:>]+):([^:>]+)(?::[^:>]+)?>", re.IGNORECASE)

# Weight-file extensions stripped from LoRA names. Not Path.stem: LoraManager
# names carry no extension, so a dot inside the name ("v1.5") must survive.
_LORA_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")

LM_LOADER = "Lora Loader (LoraManager)"
LM_TEXT_LOADER = "LoRA Text Loader (LoraManager)"
LM_STACKER = "Lora Stacker (LoraManager)"

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_ref(val) -> bool:
    """True for a graph-edge reference ``[node_id, output_index]``."""
    return isinstance(val, list) and len(val) >= 1 and isinstance(val[0], (str, int))


def _truthy(val) -> bool:
    if isinstance(val, str):
        return val.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(val)


def _flag(val):
    """``_truthy`` for a resolved flag, None when it could not be resolved."""
    return None if val is None or isinstance(val, (list, dict)) else _truthy(val)


def _is_empty_latent(class_type: str) -> bool:
    """An empty-latent node: where a generation starts (by table, or named ``Empty…Latent…``)."""
    return class_type in LATENT_SIZE_CLASSES or (class_type.startswith("Empty") and "Latent" in class_type)


def _is_size_source(class_type: str, inputs: dict) -> bool:
    """An empty-latent node whose ``width`` / ``height`` are the image size."""
    return class_type in LATENT_SIZE_CLASSES or (
        _is_empty_latent(class_type) and "width" in inputs and "height" in inputs
    )


def _sampler_kind(class_type: str, inputs: dict):
    """
    ``"ksampler"``, ``"custom"`` (SamplerCustom), ``"advanced"`` (SamplerCustomAdvanced) or
    None: by class where it is known, else by the inputs the node has.
    """
    if class_type in SAMPLER_CLASSES:
        return "ksampler"
    if class_type in CUSTOM_SAMPLERS:
        return CUSTOM_SAMPLERS[class_type]
    keys = inputs.keys()
    if ADVANCED_SIGNATURE <= keys:
        return "advanced"
    if CUSTOM_SIGNATURE <= keys:
        return "custom"
    if SAMPLER_SIGNATURE <= keys and ("seed" in keys or "noise_seed" in keys):
        return "ksampler"
    return None


def _id_key(node_id: str):
    """Sort key for node ids by number (``"9"`` before ``"12"``; ``"5:3"``, a node in a subgraph, by its parts)."""
    return [(not part.isdigit(), int(part) if part.isdigit() else 0, part) for part in str(node_id).split(":")]


def _lora_stem(name: str) -> str:
    """Base name of a LoRA without folder and weight-file extension."""
    base = re.split(r"[\\/]", name.strip())[-1]
    if base.lower().endswith(_LORA_EXTENSIONS):
        base = base.rsplit(".", 1)[0]
    return base


def _lora_tags_in_text(text: str):
    """(name, model_weight) pairs from <lora:…> tags in a text."""
    out = []
    for m in _LORA_TAG_RE.finditer(text):
        try:
            out.append((m.group(1).strip(), float(m.group(2))))
        except ValueError:
            continue
    return out


def _lm_widget_loras(loras):
    """
    Active entries of a LoraManager ``loras`` widget, or None when the widget is
    missing or unreadable. Mirrors LoraManager's own loading: only entries with
    ``active: true`` are applied, the list may be wrapped as {"__value__": [...]}.
    """
    if isinstance(loras, dict):
        loras = loras.get("__value__")
    if not isinstance(loras, list):
        return None
    out = []
    for entry in loras:
        if not isinstance(entry, dict) or not entry.get("active", False):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name.strip():
            out.append((name, entry.get("strength", 1.0)))
    return out


def _show_cache(inputs: dict, signal: str):
    """The value a display node cached from its last run (non-empty), or None."""
    for key in SHOW_CACHE_KEYS[signal]:
        val = inputs.get(key)
        if val is None or val == "" or isinstance(val, (list, dict)):
            continue
        if signal == TEXT and not isinstance(val, str):
            continue
        return val
    return None


# ---------------------------------------------------------------------------
# The walker
# ---------------------------------------------------------------------------


class Hop:
    """
    A node reached on a walk: ``node_id``, ``class_type``, ``inputs``, the output
    ``slot`` the edge came from, and ``seen``, the ids on the path so far (cycle
    guard). ``terminal``: the node is a source for the signal (nothing to follow),
    ``dead``: the signal stops here (a switch with no selected input, an unconnected
    input, an output slot that carries nothing).
    """

    __slots__ = ("node_id", "class_type", "inputs", "slot", "seen", "terminal", "dead")

    def __init__(self, node_id, class_type, inputs, slot, seen):
        self.node_id = node_id
        self.class_type = class_type
        self.inputs = inputs
        self.slot = slot
        self.seen = seen
        self.terminal = False
        self.dead = False


class Walker:
    """
    Resolves edges of one ComfyUI ``prompt`` graph. ``trace``, a list, collects the
    path taken as text lines (class names and input names, the values read).
    """

    def __init__(self, prompt: dict, trace: list | None = None):
        self.prompt = prompt
        self.trace = trace
        self._depth = 0

    # --- trace -------------------------------------------------------------

    def log(self, line: str):
        if self.trace is not None:
            self.trace.append("  " * self._depth + line)

    @contextmanager
    def nested(self):
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

    # --- the traversal -----------------------------------------------------

    def walk(self, ref, signal: str, field: str | None = None, branches: str = "first", seen=frozenset(), via=None):
        """
        Yield the nodes reached from edge ``ref``, depth first, the node itself
        before what it takes its signal from. ``branches``: ``"first"`` follows
        only the first input of a node that merges several (a model merge), so the
        walk is one path; ``"all"`` follows every one. ``field`` is the name of the
        value being resolved (scalars only).
        """
        if not _is_ref(ref):
            return
        node_id = str(ref[0])
        node = self.prompt.get(node_id)
        if node_id in seen or not isinstance(node, dict):
            return
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        slot = ref[1] if len(ref) > 1 and isinstance(ref[1], int) else 0
        hop = Hop(
            node_id,
            class_type if isinstance(class_type, str) else "",
            inputs if isinstance(inputs, dict) else {},
            slot,
            seen | {node_id},
        )
        header = len(self.trace) if self.trace is not None else 0
        self.log(f"{via or '·'} -> #{node_id} {hop.class_type}[{slot}]")
        edges = self._next(hop, signal, field)
        hop.terminal = edges is None
        hop.dead = edges == []
        if hop.terminal and self.trace is not None:
            self.trace[header] += " (end)"
        if hop.dead:
            self.log("  (the signal stops here)")
        yield hop
        for key, nxt in (edges or [])[: 1 if branches == "first" else None]:
            yield from self.walk(nxt, signal, field, branches, hop.seen, key)

    def follow(self, ref, signal: str, field: str | None = None, seen=frozenset()):
        """The source node ``ref`` leads to through nodes that pass the signal on (a Hop), or None."""
        last = None
        for last in self.walk(ref, signal, field, "first", seen):
            pass
        return last if last is not None and last.terminal else None

    # --- what passes a signal on --------------------------------------------

    def _next(self, hop: Hop, signal: str, field: str | None):
        """
        Where the signal of ``hop`` comes from: None when the node is a source (end
        of the walk), else a list of ``(input_key, edge)`` (empty = dead end).
        """
        if hop.class_type == "Reroute":
            return self._edges(hop, "*")
        if hop.class_type in SWITCHES:
            return self._switch(hop, SWITCHES[hop.class_type])
        if signal == COND:
            return self._next_cond(hop)
        if signal == IMAGE:
            return self._next_image(hop)
        if signal == SIGMAS:
            return self._edges(hop, "sigmas") if hop.class_type in SIGMAS_PASSTHROUGH else None
        if signal in (GUIDER, SAMPLER, NOISE):  # sources only: no node passes these on
            return None
        if signal == CLIP:
            return self._next_clip(hop)
        if signal == MODEL:
            return self._edges(hop, "model") or self._edges(hop, "model1", "model2") or None
        if signal == LATENT:
            return self._next_latent(hop)
        if signal == LORA_STACK:
            return self._edges(hop, "lora_stack") if hop.class_type == LM_STACKER else None
        if signal in (TEXT, SCALAR):
            if hop.class_type in VALUE_PASSTHROUGH:
                return self._edges(hop, VALUE_PASSTHROUGH[hop.class_type])
            widget = self._slot_widget(hop)
            if widget is None and signal == SCALAR and field in hop.inputs:
                # Rule of last resort for aggregator / pipe nodes: a widget named
                # like the requested field.
                widget = field
            if widget is not None:
                return self._edges(hop, widget) or None
        return None

    @staticmethod
    def _slot_widget(hop: Hop):
        """The widget an ``OUTPUT_WIDGETS`` node's output slot stands for (if it is stored)."""
        slots = OUTPUT_WIDGETS.get(hop.class_type, ())
        widget = slots[hop.slot] if 0 <= hop.slot < len(slots) else None
        return widget if widget in hop.inputs else None

    @staticmethod
    def _edges(hop: Hop, *keys):
        return [(key, hop.inputs[key]) for key in keys if _is_ref(hop.inputs.get(key))]

    def _next_cond(self, hop: Hop):
        cls = hop.class_type
        if cls in TEXT_ENCODERS or cls in COND_BLANK:
            return None
        if cls in COND_COMBINE:
            return self._edges(hop, *COND_COMBINE_KEYS)
        if cls in COND_PASSTHROUGH:
            return self._edges(hop, "conditioning")
        if all(key in hop.inputs for key in POS_NEG_SLOTS):
            key = POS_NEG_SLOTS[hop.slot] if 0 <= hop.slot < len(POS_NEG_SLOTS) else None
            return self._edges(hop, key) if key else []
        generic = self._edges(hop, "conditioning")
        if generic:
            self.log("  (generic: unknown node, its 'conditioning' input is followed)")
        return generic or None

    def _next_clip(self, hop: Hop):
        if hop.class_type in CLIP_SOURCES:
            return None
        edges = self._edges(hop, "clip")
        if edges and hop.class_type not in CLIP_PASSTHROUGH:
            self.log("  (generic: unknown node, its 'clip' input is followed)")
        return edges or None

    def _next_latent(self, hop: Hop):
        if _is_size_source(hop.class_type, hop.inputs):
            return None
        for key in LATENT_INPUTS:
            if _is_ref(hop.inputs.get(key)):
                return self._edges(hop, key)
        return None

    def _next_image(self, hop: Hop):
        """
        The way back from a picture to where it was made: through image nodes (``IMAGE_INPUTS``),
        decoders, encoders and samplers (``LATENT_INPUTS``), to the first node that has none of
        them connected (an empty latent, a loaded image). Every node takes its first connected input.
        """
        if _is_empty_latent(hop.class_type):
            return None
        for key in IMAGE_INPUTS + LATENT_INPUTS:
            if _is_ref(hop.inputs.get(key)):
                return self._edges(hop, key)
        return None

    def _switch(self, hop: Hop, rule: Switch):
        """The input a known switch passes on: ``[(key, edge)]``, or ``[]`` when none is selected."""
        inputs = hop.inputs
        if rule.kind in ("bool", "soft"):
            true_edge, false_edge = self._edges(hop, "on_true"), self._edges(hop, "on_false")
            if rule.kind == "soft" and bool(true_edge) != bool(false_edge):
                return true_edge or false_edge
            flag = _flag(self.scalar(inputs.get("switch"), "switch", seen=hop.seen))
            if flag is None:
                self.log("  (switch flag not resolved: nothing selected)")
                return []
            self.log(f"  (switch={flag})")
            return true_edge if flag else false_edge
        if rule.kind == "first":
            slots = sorted(
                (int(key[4:]), key)
                for key, val in inputs.items()
                if re.fullmatch(r"any_\d+", key) and _is_ref(val)
            )
            return self._edges(hop, slots[0][1]) if slots else []
        try:
            number = int(self.scalar(inputs.get(rule.select), rule.select, seen=hop.seen))
        except (TypeError, ValueError):
            self.log(f"  ('{rule.select}' not resolved: nothing selected)")
            return []
        self.log(f"  ({rule.select}={number})")
        if rule.kind == "inversed":
            return self._edges(hop, rule.input) if hop.slot == number - rule.base else []
        for prefix in rule.prefixes:
            edges = self._edges(hop, f"{prefix}{number}")
            if edges:
                return edges
        return []

    # --- scalars -----------------------------------------------------------

    def scalar(self, val, field: str, size: bool = False, seen=frozenset()):
        """
        A scalar widget value: ``val`` as stored in an input, a literal or a graph
        edge. ``field`` is the name of the input being resolved ("steps", "cfg", …)
        and decides which widgets of a source node answer (see ``_value_keys``).
        ``size``: the value is a width/height wired into an empty-latent node, see
        ``_size_of``.
        """
        if val is None:
            return None
        if not _is_ref(val):
            return val if not isinstance(val, list) else None
        hop = self.follow(val, SCALAR, field, seen)
        if hop is None:
            return None
        result = self._scalar_of(hop, field, size)
        if self.trace is not None and result is not None:
            self.log(f"  = {result!r}")
        return result

    def _scalar_of(self, hop: Hop, field: str, size: bool):
        inputs = hop.inputs
        key = self._slot_widget(hop) or (field if field in inputs else None)
        if key is not None:
            val = inputs[key]
            return val if not isinstance(val, list) else None
        if hop.class_type in SHOW_PASSTHROUGH:
            return self._show_value(hop, SCALAR, field, size)
        if size and field in ("width", "height"):
            found = self._size_of(hop, field)
            if found is not None:
                return found
        for key in _value_keys(field):
            val = inputs.get(key)
            if val is None:
                continue
            if not isinstance(val, list):
                return val
            found = self.scalar(val, field, size, hop.seen)
            if found is not None:
                return found
        return None

    def _show_value(self, hop: Hop, signal: str, field: str | None = None, size: bool = False):
        """A display node's value: what its passthrough edge resolves to, else its cached widget."""
        for key in SHOW_FOLLOW_KEYS:
            edge = hop.inputs.get(key)
            if not _is_ref(edge):
                continue
            self.log(f"  (display node: following '{key}')")
            with self.nested():
                value = self.text(edge, hop.seen) if signal == TEXT else self.scalar(edge, field, size, hop.seen)
            if value not in (None, ""):
                return value
            self.log("  (source not resolved: the cached widget is used)")
            break
        cached = _show_cache(hop.inputs, signal)
        if cached is not None:
            self.log("  = cached widget")
        return cached

    def _size_of(self, hop: Hop, field: str):
        """
        Width or height from a size picker, for nodes at the end of the width/height
        edge of an empty-latent node: the first "W×H" in a string widget (a preset
        label such as "Portrait 2:3 · 1152×1728 · 1.9 MP"; only literal strings,
        width first), else the widget whose name contains the field
        (``custom_width`` / ``custom_height``).
        """
        for val in hop.inputs.values():
            match = SIZE_IN_LABEL.search(val) if isinstance(val, str) else None
            if match:
                self.log("  (size read from a 'W×H' label)")
                return int(match.group(1 if field == "width" else 2))
        for key, val in hop.inputs.items():
            if field in key:
                found = self.scalar(val, key, True, hop.seen)
                if found is not None:
                    return found
        return None

    # --- text ----------------------------------------------------------------

    def text(self, ref, seen=frozenset()) -> str:
        """Resolve a string input (a literal or a graph edge) to its text."""
        if isinstance(ref, str):
            return ref
        if isinstance(ref, (int, float)):
            return str(ref)
        hop = self.follow(ref, TEXT, None, seen)
        if hop is None:
            return ""
        with self.nested():
            return self._text_of(hop)

    def _text_of(self, hop: Hop) -> str:
        cls, inputs = hop.class_type, hop.inputs
        scalar = lambda key: self.scalar(inputs.get(key), key, seen=hop.seen)  # noqa: E731

        if cls in SHOW_PASSTHROUGH:
            value = self._show_value(hop, TEXT)
            if value is not None:
                return value

        widget = self._slot_widget(hop)
        if widget is not None:
            return self.text(inputs[widget], hop.seen)

        # ImpactWildcardProcessor: 'populated_text' holds the already-resolved
        # wildcard text (when mode='reproduce').
        if cls == "ImpactWildcardProcessor":
            populated = inputs.get("populated_text", "")
            if _is_ref(populated):
                return self.text(populated, hop.seen)
            return str(populated) if populated else ""

        # RegexReplace: follow the string input, then apply the regex substitution.
        # The pattern, replacement, count and flags may be wired from other nodes.
        if cls == "RegexReplace":
            text = self.text(inputs.get("string"), hop.seen)
            if text:
                pattern = self.text(inputs.get("regex_pattern", ""), hop.seen)
                replace = self.text(inputs.get("replace", ""), hop.seen)
                try:
                    count = int(scalar("count") or 0)
                except (TypeError, ValueError):
                    count = 0
                flags = 0
                for key, flag in (("case_insensitive", re.IGNORECASE), ("multiline", re.MULTILINE), ("dotall", re.DOTALL)):
                    if _truthy(scalar(key)):
                        flags |= flag
                try:
                    text = re.sub(pattern, replace, text, count=count, flags=flags)
                except (re.error, ValueError):
                    pass
            return text

        # StringFunction|pysssss: same logic as the node's ``exec``.
        if cls in ("StringFunction|pysssss", "StringFunction"):
            return self._string_function(hop)

        # StringConcatenate (comfy-core): ``delimiter.join((string_a, string_b))``;
        # the delimiter widget defaults to "" and empty parts are kept.
        if cls == "StringConcatenate":
            delimiter = inputs.get("delimiter", "")
            if _is_ref(delimiter):
                delimiter = self.text(delimiter, hop.seen)
            part_a = self.text(inputs.get("string_a"), hop.seen)
            part_b = self.text(inputs.get("string_b"), hop.seen)
            return str(delimiter if delimiter is not None else "").join((part_a, part_b))

        # Nodes that output one of their text inputs unchanged (see TEXT_PASSTHROUGH).
        if cls in TEXT_PASSTHROUGH:
            return self.text(inputs.get(TEXT_PASSTHROUGH[cls]), hop.seen)

        # Primitive / ShowText / Note and others: a text stored in a widget.
        for key in TEXT_KEYS:
            val = inputs.get(key)
            if isinstance(val, list):
                return self.text(val, hop.seen)
            if val is not None and not isinstance(val, dict):
                self.log(f"  = widget '{key}'")
                return str(val)
        return ""

    def _string_function(self, hop: Hop) -> str:
        """``StringFunction|pysssss``: append or replace, then optionally tidy tags."""
        inputs = hop.inputs
        action = inputs.get("action", "append")
        tidy = inputs.get("tidy_tags", "yes") == "yes"
        text_a, text_b, text_c = (self.text(inputs.get(k), hop.seen) for k in ("text_a", "text_b", "text_c"))
        if action == "append":
            out = (", " if tidy else "").join(t for t in (text_a, text_b, text_c) if t)
        elif text_b.startswith("/") and text_b.endswith("/"):
            try:
                out = re.sub(text_b[1:-1], text_c, text_a)
            except re.error:
                out = text_a
        else:
            out = text_a.replace(text_b, text_c)
        if tidy:
            out = re.sub(r"\s{2,}", " ", out)
            out = out.replace(" ,", ",")
            out = re.sub(r",{2,}", ",", out)
            out = out.strip()
        return out

    # --- conditioning --------------------------------------------------------

    def cond_text(self, ref, seen=frozenset()) -> str:
        """The prompt text a conditioning edge was encoded from (merged conditioning: all texts, joined)."""
        parts = []
        for hop in self.walk(ref, COND, None, "all", seen):
            if hop.class_type not in TEXT_ENCODERS:
                continue
            with self.nested():
                text = self._encode_text(hop)
            if text:
                parts.append(text)
        return ", ".join(parts)

    def _encode_text(self, hop: Hop) -> str:
        parts = []
        for key in TEXT_ENCODERS[hop.class_type]:
            val = hop.inputs.get(key)
            if val is None:
                continue
            text = self.text(val, hop.seen)
            if text and text not in parts:
                parts.append(text)
        return "\n".join(parts)

    # --- model ---------------------------------------------------------------

    def model_name(self, ref) -> str:
        """File name of the model at the end of the model chain: the first path through a merge."""
        for hop in self.walk(ref, MODEL, None, "first"):
            widgets = MODEL_LOADERS.get(hop.class_type)
            if widgets is None and hop.terminal:
                widgets = MODEL_SOURCE_WIDGETS
            if widgets is None:
                continue
            for widget in widgets:
                name = self.scalar(hop.inputs.get(widget), widget, seen=hop.seen)
                if isinstance(name, str) and name:
                    return name
            return ""
        return ""

    def loras(self, ref):
        """
        Every LoRA applied along the model chain (both inputs of a merge) as
        ``(name, weight)``, de-duplicated by file stem (first weight wins).
        """
        found = []
        for hop in self.walk(ref, MODEL, None, "all"):
            found.extend(self._node_loras(hop))
        result = []
        seen_stems = set()
        for name, weight in found:
            stem = _lora_stem(name).lower()
            if stem not in seen_stems:
                seen_stems.add(stem)
                result.append((name, weight))
        return result

    def _node_loras(self, hop: Hop):
        """(lora_name, weight) pairs configured directly on one node (the LoRA-loader variants)."""
        cls, inputs = hop.class_type, hop.inputs
        scalar = lambda key: self.scalar(inputs.get(key), key, seen=hop.seen)  # noqa: E731
        out = []

        # LoraManager loader: applies its incoming lora_stack, then the active
        # entries of its ``loras`` widget. The ``text`` widget is only an editor
        # mirror that still lists deactivated LoRAs, so it is read only when the
        # structured list is missing.
        if cls == LM_LOADER:
            out.extend(self._stack_loras(inputs.get("lora_stack"), hop.seen))
            widget = _lm_widget_loras(inputs.get("loras"))
            if widget is None:
                text = inputs.get("text")
                widget = _lora_tags_in_text(text) if isinstance(text, str) else []
            out.extend(widget)

        # LoraManager text loader: lora_stack, then the <lora:…> tags of lora_syntax.
        elif cls == LM_TEXT_LOADER:
            out.extend(self._stack_loras(inputs.get("lora_stack"), hop.seen))
            syntax = inputs.get("lora_syntax")
            if _is_ref(syntax):
                syntax = self.text(syntax, hop.seen)
            if isinstance(syntax, str):
                out.extend(_lora_tags_in_text(syntax))

        # rgthree Power Lora Loader: one entry per ``lora_N`` key, each a dict
        # {"on": bool, "lora": name, "strength": weight}.
        elif cls == "Power Lora Loader (rgthree)":
            for key, val in inputs.items():
                if not re.fullmatch(r"lora_\d+", key) or not isinstance(val, dict):
                    continue
                name = val.get("lora")
                if val.get("on", True) and isinstance(name, str) and name.lower() != "none":
                    out.append((name, val.get("strength", 1.0)))

        # Core LoraLoader / LoraLoaderModelOnly (strengths may be wired from other nodes).
        elif cls in ("LoraLoader", "LoraLoaderModelOnly"):
            name = scalar("lora_name")
            if isinstance(name, str) and name.lower() != "none":
                weight = scalar("strength_model")
                if weight is None:
                    weight = scalar("strength")
                if weight is None:
                    weight = 1.0
                # Strength 0 on model and clip = loader switched off (the idle
                # slot in img2prompt's gen graph): not applied, so not listed.
                clip = scalar("strength_clip")
                if clip is None:
                    clip = 0
                if not (weight == 0 and clip == 0):
                    out.append((name, weight))

        # rgthree Lora Loader Stack: lora_1..N / strength_1..N, "None" = empty slot.
        elif cls == "Lora Loader Stack (rgthree)":
            i = 1
            while f"lora_{i}" in inputs:
                name = scalar(f"lora_{i}")
                if isinstance(name, str) and name.lower() != "none":
                    weight = scalar(f"strength_{i}")
                    out.append((name, 1.0 if weight is None else weight))
                i += 1

        for name, weight in out:
            self.log(f"  = LoRA {name!r} {weight!r}")
        return out

    def _stack_loras(self, stack_ref, seen):
        """LoRAs of a LoraManager ``lora_stack`` link: the chained Stacker nodes, the innermost stack first."""
        stackers = [h for h in self.walk(stack_ref, LORA_STACK, None, "first", seen) if h.class_type == LM_STACKER]
        out = []
        for hop in reversed(stackers):
            out.extend(_lm_widget_loras(hop.inputs.get("loras")) or [])
        return out

    # --- clip skip -----------------------------------------------------------

    def clip_skip(self, ref):
        """
        The clip skip of the first text encoder the conditioning edge ``ref`` reaches: from its
        ``clip`` input back through the nodes that pass CLIP on (``CLIP_PASSTHROUGH``, any node
        with a ``clip`` input) to the first ``CLIP_SOURCES`` node, whose layer is negated
        (-2 gives 2). None when the walk ends at anything else (a loader, say), the layer is
        not resolved or it is not a negative whole number.
        """
        encoder = next((hop for hop in self.walk(ref, COND, None, "all") if hop.class_type in TEXT_ENCODERS), None)
        if encoder is None:
            return None
        self.log(f"  (first text encoder: #{encoder.node_id}, its 'clip' input is followed)")
        source = self.follow(encoder.inputs.get("clip"), CLIP, None, encoder.seen)
        widget = CLIP_SOURCES.get(source.class_type) if source else None
        if widget is None:
            return None
        layer = self.scalar(source.inputs.get(widget), widget, seen=source.seen)
        try:
            number = float(layer)
        except (TypeError, ValueError):
            return None
        skip = -int(number) if number.is_integer() else 0
        return skip if skip >= 1 else None

    # --- latent size ---------------------------------------------------------

    def latent_size(self, ref):
        """(width, height) if the latent originates from an empty-latent node, otherwise (None, None)."""
        hop = self.follow(ref, LATENT)
        if hop is None or not _is_size_source(hop.class_type, hop.inputs):
            return None, None
        width = self.scalar(hop.inputs.get("width"), "width", size=True, seen=hop.seen)
        height = self.scalar(hop.inputs.get("height"), "height", size=True, seen=hop.seen)
        return width, height


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def _safe(func, default):
    """Run one field's extraction; a field that fails gets its "unresolved" value."""
    try:
        return func()
    except Exception:  # noqa: BLE001 - one broken node must not cost the other fields
        return default


def _find_output_sampler(walker: Walker):
    """
    The sampler the saved pictures come from, as ``(node_id, node, reason)``, or None.

    Starts at the output nodes: save nodes first (known classes, then any node with an
    ``images`` input that nothing uses), preview nodes only when no save node leads to a
    sampler; by node id within each group. The first output node with a sampler behind it
    decides. From there the picture is followed back (decode, upscale, encode, …) to the
    latent it started from. Of the samplers on that way, the first pass counts, as in A1111's
    hires fix: the one whose latent comes from an empty-latent node. Without such a start
    (img2img), the sampler nearest the output counts.
    """
    prompt = walker.prompt
    nodes = {
        nid: node
        for nid, node in prompt.items()
        if isinstance(node, dict) and isinstance(node.get("class_type"), str) and isinstance(node.get("inputs"), dict)
    }
    used = {str(val[0]) for node in nodes.values() for val in node["inputs"].values() if _is_ref(val)}
    saves, others, previews = [], [], []
    for nid, node in sorted(nodes.items(), key=lambda item: _id_key(item[0])):
        cls = node["class_type"]
        if cls in SAVE_CLASSES:
            saves.append((nid, node, SAVE_CLASSES[cls]))
        elif _is_ref(node["inputs"].get("images")):
            if cls in PREVIEW_CLASSES:
                previews.append((nid, node, "images"))
            elif nid not in used:
                others.append((nid, node, "images"))
    for nid, node, key in saves + others + previews:
        walker.log(f"output #{nid} {node['class_type']}")
        samplers, start = [], None
        for hop in walker.walk(node["inputs"].get(key), IMAGE):
            if _sampler_kind(hop.class_type, hop.inputs):
                samplers.append(hop)
            start = hop
        if not samplers:
            continue
        if start.terminal and _is_empty_latent(start.class_type):
            return samplers[-1].node_id, prompt[samplers[-1].node_id], (
                f"first pass: its latent starts at #{start.node_id} {start.class_type}"
            )
        return samplers[0].node_id, prompt[samplers[0].node_id], (
            "nearest to the output: no empty latent at the start of its way back"
        )
    return None


def _find_sampler(prompt: dict):
    """
    The sampler when no output node leads to one, as ``(node_id, node)``, or None: the
    first of the known KSampler classes in the graph (``KSampler`` and ``KSamplerAdvanced``
    before the variants), else the first node by id that has a sampler's inputs.
    """
    sampler_nodes = [
        (nid, node)
        for nid, node in prompt.items()
        if isinstance(node, dict)
        and isinstance(node.get("class_type"), str)
        and node["class_type"] in SAMPLER_CLASSES
    ]
    if sampler_nodes:
        # Prefer KSampler / KSamplerAdvanced over specialised variants
        return next(
            ((nid, n) for nid, n in sampler_nodes if n.get("class_type") in ("KSampler", "KSamplerAdvanced")),
            sampler_nodes[0],
        )
    for nid, node in sorted(prompt.items(), key=lambda item: _id_key(item[0])):
        if isinstance(node, dict) and isinstance(node.get("class_type"), str) and isinstance(node.get("inputs"), dict):
            if _sampler_kind(node["class_type"], node["inputs"]):
                return nid, node
    return None


def _ksampler_sources(inputs: dict) -> dict:
    """
    Where a sampler keeps what the extraction needs: edges for ``positive``,
    ``negative``, ``model`` and ``latent``, and for each scalar its stored value (a
    literal or an edge) together with the name of the field. A KSampler has all of
    them as inputs; the samplers below take them from the nodes they are wired to, but
    return the same: the values are those nodes' inputs, resolved to the edge or literal
    there.
    """
    # KSamplerAdvanced and its variants name the seed ``noise_seed``.
    seed_key = "noise_seed" if "noise_seed" in inputs else "seed"
    return {
        "positive": inputs.get("positive"),
        "negative": inputs.get("negative"),
        "model": inputs.get("model"),
        "latent": inputs.get("latent_image"),
        "seed": (inputs.get(seed_key), seed_key),
        **{key: (inputs.get(key), key) for key in ("steps", "cfg", "sampler_name", "scheduler")},
    }


def _sampler_name_source(walker: Walker, ref):
    """``sampler_name`` of a ``sampler`` edge (KSamplerSelect and the like), as a (value, field) pair."""
    hop = walker.follow(ref, SAMPLER)
    name = SAMPLER_NODES.get(hop.class_type, False) if hop else False
    if name is False:
        return None, "sampler_name"
    walker.log(f"  sampler: #{hop.node_id} {hop.class_type}")
    if name is None:
        return hop.inputs.get("sampler_name"), "sampler_name"
    if hop.class_type in SAMPLER_NOISE_DEVICE:
        device = walker.scalar(hop.inputs.get("noise_device"), "noise_device", seen=hop.seen)
        if device not in ("cpu", "gpu"):
            return None, "sampler_name"
        name += "_gpu" if device == "gpu" else ""
    return name, "sampler_name"


def _sigmas_sources(walker: Walker, ref) -> dict:
    """
    Steps and scheduler of a ``sigmas`` edge as (value, field) pairs, and the scheduler's
    ``model`` edge. The scheduler is the node the sigmas come from; passthrough nodes in
    between that change the number of steps leave the steps empty.
    """
    hops = list(walker.walk(ref, SIGMAS))
    source = hops[-1] if hops and hops[-1].terminal and hops[-1].class_type in SCHEDULERS else None
    if source is None:
        return {"steps": (None, "steps"), "scheduler": (None, "scheduler"), "model": None}
    walker.log(f"  sigmas: #{source.node_id} {source.class_type}")
    fixed = SCHEDULERS[source.class_type]
    resized = any(hop.class_type in SIGMAS_RESIZERS for hop in hops)
    return {
        "steps": (None if resized else source.inputs.get("steps"), "steps"),
        "scheduler": (source.inputs.get("scheduler") if fixed is None else fixed, "scheduler"),
        "model": source.inputs.get("model"),
    }


def _custom_sources(walker: Walker, inputs: dict) -> dict:
    """``SamplerCustom``: ``sampler`` and ``sigmas`` are nodes, the rest are its own inputs."""
    sigmas = _sigmas_sources(walker, inputs.get("sigmas"))
    seed = (inputs.get("noise_seed"), "noise_seed")
    if _flag(walker.scalar(inputs.get("add_noise"), "add_noise")) is False:
        seed = (None, "noise_seed")  # without noise the seed does nothing
    return {
        "positive": inputs.get("positive"),
        "negative": inputs.get("negative"),
        "model": inputs.get("model"),
        "latent": inputs.get("latent_image"),
        "seed": seed,
        "steps": sigmas["steps"],
        "cfg": (inputs.get("cfg"), "cfg"),
        "sampler_name": _sampler_name_source(walker, inputs.get("sampler")),
        "scheduler": sigmas["scheduler"],
    }


def _advanced_sources(walker: Walker, inputs: dict) -> dict:
    """``SamplerCustomAdvanced``: noise, guider, sampler and sigmas are nodes."""
    sigmas = _sigmas_sources(walker, inputs.get("sigmas"))
    guider = walker.follow(inputs.get("guider"), GUIDER)
    rule = GUIDERS.get(guider.class_type) if guider else None
    given = guider.inputs if rule else {}
    if rule:
        walker.log(f"  guider: #{guider.node_id} {guider.class_type}")
    noise = walker.follow(inputs.get("noise"), NOISE)
    seed_key = NOISE_SEEDS.get(noise.class_type) if noise else None
    if noise:
        walker.log(f"  noise: #{noise.node_id} {noise.class_type}")
    return {
        "positive": given.get(rule.positive) if rule and rule.positive else None,
        "negative": given.get(rule.negative) if rule and rule.negative else None,
        # the guider's model; without a known guider the scheduler's, if it has one
        "model": given.get("model") if rule else sigmas["model"],
        "latent": inputs.get("latent_image"),
        "seed": (noise.inputs.get(seed_key) if seed_key else None, "noise_seed"),
        "steps": sigmas["steps"],
        "cfg": (given.get(rule.cfg) if rule and rule.cfg else None, "cfg"),
        "sampler_name": _sampler_name_source(walker, inputs.get("sampler")),
        "scheduler": sigmas["scheduler"],
    }


def _sampler_sources(walker: Walker, node: dict) -> dict:
    """What ``_extract_fields`` reads, for a sampler node of any kind (see ``_sampler_kind``)."""
    inputs = node["inputs"] if isinstance(node.get("inputs"), dict) else {}
    kind = _sampler_kind(node["class_type"], inputs)
    if kind == "custom":
        return _custom_sources(walker, inputs)
    if kind == "advanced":
        return _advanced_sources(walker, inputs)
    return _ksampler_sources(inputs)


def _extract_fields(walker: Walker, src: dict) -> dict:
    """The result dict for a sampler described by ``src`` (see ``_ksampler_sources``)."""

    def field(name, func, default):
        walker.log(f"[{name}]")
        value = _safe(func, default)
        if walker.trace is not None:
            walker.log(f"= {value!r}")
        return value

    def scalar(name):
        value, key = src[name]
        return field(name, lambda: walker.scalar(value, key), None)

    walker.log("[width, height]")
    width, height = _safe(lambda: walker.latent_size(src["latent"]), (None, None))
    walker.log(f"= {width!r} x {height!r}")

    return {
        "positive": field("positive", lambda: walker.cond_text(src["positive"]), ""),
        "negative": field("negative", lambda: walker.cond_text(src["negative"]), ""),
        "seed": scalar("seed"),
        "steps": scalar("steps"),
        "cfg": scalar("cfg"),
        "sampler_name": scalar("sampler_name"),
        "scheduler": scalar("scheduler"),
        "model_name": field("model_name", lambda: walker.model_name(src["model"]), ""),
        "width": width,
        "height": height,
        "loras": field("loras", lambda: walker.loras(src["model"]), []),
        "clip_skip": field("clip_skip", lambda: walker.clip_skip(src["positive"]), None),
    }


def _extract(prompt, trace: list | None):
    if not isinstance(prompt, dict):
        return None
    walker = Walker(prompt, trace)
    walker.log("[sampler]")
    found = _safe(lambda: _find_output_sampler(walker), None)
    if found is None:
        fallback = _find_sampler(prompt)
        if fallback is None:
            return None
        found = (*fallback, "no output node leads to a sampler")
    nid, node, reason = found
    walker.log(f"sampler #{nid} {node['class_type']} ({reason})")
    src = _safe(lambda: _sampler_sources(walker, node), None)
    if src is None:  # the sampler's inputs could not be read: as if there were none
        walker.log("(the sampler's inputs could not be read)")
        return None
    return _extract_fields(walker, src)


def extract_from_prompt(prompt: dict) -> dict | None:
    """
    Find the sampler that made the saved picture in the ComfyUI graph and extract all
    relevant generation parameters. Returns None when no sampler node is found or the
    input is not a prompt dict. Which sampler counts: see ``_find_output_sampler``
    (without an output node leading to one, ``_find_sampler``).

    ``prompt`` is the parsed ComfyUI API prompt (node id -> {"class_type", "inputs"}).
    The returned dict always has these keys; a value that cannot be resolved (or
    whose extraction fails) is "" (text fields), None (scalars) or [] (loras):

    - ``positive``, ``negative``: prompt text (str)
    - ``seed``, ``steps``, ``cfg``: as stored in the graph (int / float, rarely str)
    - ``sampler_name``, ``scheduler``: str or None
    - ``model_name``: checkpoint / unet file name as in the graph (str)
    - ``width``, ``height``: latent size, None if not from an empty-latent node
    - ``loras``: list of (name, weight) tuples on the sampler's model chain
    - ``clip_skip``: int >= 1 from the ``CLIPSetLastLayer`` on the clip path of the positive
      prompt's first text encoder (-2 is 2), None without one
    """
    return _extract(prompt, None)


def trace_from_prompt(prompt: dict):
    """``(info, lines)``: ``extract_from_prompt`` and the path it took, as text lines."""
    trace: list = []
    return _extract(prompt, trace), trace


# ---------------------------------------------------------------------------
# A1111 output formatting
# ---------------------------------------------------------------------------


def model_basename(model_name) -> str:
    """Checkpoint name without folder (``/`` or ``\\``) and final extension."""
    if not isinstance(model_name, str):
        return ""
    return PurePosixPath(model_name.replace("\\", "/")).stem


def lora_entries(info: dict):
    """
    The LoRAs of ``info`` as ``(stem, weight)`` pairs, ``weight`` a float: the
    name without folder and weight-file extension, the weight formatted with
    ``%g`` (1 when it is not a number). The single source for the prompt tags
    and for the IIB adapter's LoRA list.
    """
    out = []
    for name, weight in info.get("loras") or []:
        try:
            w = float(f"{float(weight):g}")
        except (TypeError, ValueError):
            w = 1.0
        out.append((_lora_stem(name), w))
    return out


def _lora_tags(info: dict) -> str:
    """Render collected LoRAs as A1111 ``<lora:name:weight>`` tags. IIB (and its
    webui "Resources" panel) reads LoRAs from these prompt tags, so emitting them
    surfaces the graph's LoRAs the same way A1111 metadata would."""
    return " ".join(f"<lora:{stem}:{w:g}>" for stem, w in lora_entries(info))


def a1111_prompt_text(info: dict, lora_separator: str = " ") -> str:
    """
    Positive prompt (with LoRA tags) and ``Negative prompt:`` line, no parameters
    line. ``lora_separator`` goes between the positive text and the LoRA tags.
    """
    lines = []
    positive = (info.get("positive") or "").strip()
    lora_tags = _lora_tags(info)
    if lora_tags:
        positive = f"{positive}{lora_separator}{lora_tags}" if positive else lora_tags
    if positive:
        lines.append(positive)

    neg = (info.get("negative") or "").strip()
    if neg:
        lines.append(f"Negative prompt: {neg}")
    return "\n".join(lines)


def a1111_params(info: dict, resources: dict | None = None):
    """The A1111 parameters line as ordered ``(key, value)`` string pairs (the
    keys ``Steps``, ``Sampler``, ``Schedule type`` (only with a sampler), ``CFG scale``,
    ``Seed``, ``Size``, ``Model``). ``resources`` (see ``image_saver_resources``)
    adds ``Model hash`` before ``Model`` and ``Hashes``, ``Lora hashes`` and
    ``Civitai resources`` after it; the values are plain strings, JSON compact."""
    parts = []
    if info.get("steps") is not None:
        parts.append(("Steps", f"{info['steps']}"))
    if info.get("sampler_name"):
        parts.append(("Sampler", f"{info['sampler_name']}"))
        if info.get("scheduler"):
            parts.append(("Schedule type", f"{info['scheduler']}"))
    if info.get("cfg") is not None:
        parts.append(("CFG scale", f"{info['cfg']}"))
    if info.get("seed") is not None:
        parts.append(("Seed", f"{info['seed']}"))
    if info.get("width") and info.get("height"):
        parts.append(("Size", f"{info['width']}x{info['height']}"))
    res = _resource_values(resources)
    if "Model hash" in res:
        parts.append(("Model hash", res["Model hash"]))
    model = model_basename(info.get("model_name"))
    if model:
        parts.append(("Model", model))
    parts.extend((key, res[key]) for key in _QUOTED_KEYS if key in res)
    return parts


def format_a1111(info: dict, resources: dict | None = None) -> str:
    """Format extracted generation fields as an A1111-style parameters string.
    The values of ``Hashes``, ``Lora hashes`` and ``Civitai resources`` are written
    as JSON strings, so that readers splitting the line at commas keep them whole."""
    lines = []
    text = a1111_prompt_text(info)
    if text:
        lines.append(text)
    params = a1111_params(info, resources)
    if params:
        lines.append(", ".join(
            f"{k}: {json.dumps(v, ensure_ascii=False) if k in _QUOTED_KEYS else v}" for k, v in params
        ))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resources from an A1111 ``parameters`` chunk (Image Saver)
# ---------------------------------------------------------------------------
#
# Some ComfyUI images (node "Image Saver", comfyui-image-saver) carry a
# ``parameters`` text chunk next to the ``prompt`` graph. The graph stays the source
# of the generation fields; only the resources are taken from the text: model
# hash, hashes of model / LoRAs / embeddings and the Civitai resource list.

_QUOTED_KEYS = ("Hashes", "Lora hashes", "Civitai resources")
_RESOURCE_KEYS = {
    "model hash": "model_hash",
    "hashes": "hashes",
    "lora hashes": "lora_hashes",
    "civitai resources": "civitai_resources",
}
_RE_SETTING_KEY = re.compile(r"\s*([\w ]+):[ \t]*")
_RE_NEXT_SETTING = re.compile(r",\s*[\w ]+:(?:\s|$)")
_RE_VALUE_END = re.compile(r"\s*(?:,|$)")
_RE_NAME_HASH = re.compile(r"\s*([^,]+?):\s*([0-9A-Za-z]+)\s*(?:,|$)")
_JSON = json.JSONDecoder()


def _compact(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _settings_line(text: str):
    """The last line that starts with ``Steps: ``, else None."""
    for line in reversed(text.splitlines()):
        if line.startswith("Steps: "):
            return line
    return None


def _parse_settings(line: str) -> dict:
    """
    The ``Key: value`` pairs of an A1111 settings line, keys lower-case, the first
    of a repeated key wins. A value starting with ``{``, ``[`` or ``"`` is read as
    JSON (so commas inside it do not split it), if it is valid and followed by a
    comma or the end of the line; any other value ends at the next ``, Key: ``.
    JSON values come back decoded, the rest as stripped strings.
    """
    out = {}
    pos, end = 0, len(line)
    while pos < end:
        key = _RE_SETTING_KEY.match(line, pos)
        if not key:
            comma = line.find(",", pos)
            if comma < 0:
                break
            pos = comma + 1
            continue
        pos = key.end()
        value, after = None, None
        if pos < end and line[pos] in '{["':
            try:
                value, after = _JSON.raw_decode(line, pos)
            except (ValueError, RecursionError):  # deep nesting: RecursionError up to 3.11
                after = None
            tail = _RE_VALUE_END.match(line, after) if after is not None else None
            if tail:
                pos = tail.end()
            else:
                after = None
        if after is None:
            nxt = _RE_NEXT_SETTING.search(line, pos)
            stop = nxt.start() if nxt else end
            value = line[pos:stop].strip()
            pos = stop + 1
        out.setdefault(key.group(1).strip().lower(), value)
    return out


def _json_value(value):
    """``value`` itself, or what it decodes to when it is a string holding JSON."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, RecursionError):
            return None
    return value


def image_saver_resources(parameters) -> dict:
    """
    Resources from the A1111 ``parameters`` text of an image, as a dict that has
    only the entries found (``{}`` for no text, no settings line or nothing usable;
    never raises on malformed text):

    - ``model_hash``: str
    - ``hashes``: dict, name -> hash (``Hashes``: ``model``, ``LORA:<name>``, ``embed:<name>``)
    - ``lora_hashes``: dict, name -> hash (``Lora hashes: "name: hash, …"``)
    - ``civitai_resources``: list of dicts (``Civitai resources``), any element form

    Both forms of Image Saver are read: the older one has only ``Civitai resources``
    (JSON array, not quoted), the newer one all four, ``Hashes`` as compact JSON and
    ``Lora hashes`` quoted. Prompt, sampler, seed and the other settings are ignored.
    """
    if not isinstance(parameters, str):
        return {}
    try:
        line = _settings_line(parameters)
        settings = _parse_settings(line) if line else {}
        found = {name: settings[key] for key, name in _RESOURCE_KEYS.items() if key in settings}
        out = {}
        model_hash = found.get("model_hash")
        if isinstance(model_hash, str) and model_hash:
            out["model_hash"] = model_hash
        hashes = _json_value(found["hashes"]) if "hashes" in found else None
        if isinstance(hashes, dict):
            hashes = {str(k): str(v) for k, v in hashes.items() if isinstance(v, (str, int)) and str(v)}
            if hashes:
                out["hashes"] = hashes
        lora_hashes = found.get("lora_hashes")
        if isinstance(lora_hashes, str):
            pairs = {name.strip(): value for name, value in _RE_NAME_HASH.findall(lora_hashes)}
            if pairs:
                out["lora_hashes"] = pairs
        resources = _json_value(found["civitai_resources"]) if "civitai_resources" in found else None
        if isinstance(resources, list):
            resources = [item for item in resources if isinstance(item, dict) and item]
            if resources:
                out["civitai_resources"] = resources
        return out
    except Exception:  # noqa: BLE001 - a broken chunk must never fail the caller
        return {}


def _resource_values(resources) -> dict:
    """``image_saver_resources`` output as ``{A1111 key: string}``, plain (unquoted)."""
    if not isinstance(resources, dict):
        return {}
    out = {}
    if resources.get("model_hash"):
        out["Model hash"] = str(resources["model_hash"])
    if resources.get("hashes"):
        out["Hashes"] = _compact(resources["hashes"])
    if resources.get("lora_hashes"):
        out["Lora hashes"] = ", ".join(f"{name}: {value}" for name, value in resources["lora_hashes"].items())
    if resources.get("civitai_resources"):
        out["Civitai resources"] = _compact(resources["civitai_resources"])
    return out


# ---------------------------------------------------------------------------
# Command line: python3 graphwalk.py <image.png | prompt.json> [--trace]
# ---------------------------------------------------------------------------

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_text_chunks(data: bytes):
    """``{keyword: text}`` of the tEXt, zTXt and iTXt chunks of a PNG (standard library only)."""
    chunks = {}
    pos = len(PNG_SIGNATURE)
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length  # length, type, data, crc
        try:
            if kind == b"tEXt":
                key, _, text = body.partition(b"\x00")
                chunks[key.decode("latin-1")] = text.decode("latin-1")
            elif kind == b"zTXt":
                key, _, rest = body.partition(b"\x00")
                chunks[key.decode("latin-1")] = zlib.decompress(rest[1:]).decode("latin-1")
            elif kind == b"iTXt":
                key, _, rest = body.partition(b"\x00")
                compressed = rest[0]
                _lang, _, rest = rest[2:].partition(b"\x00")
                _translated, _, text = rest.partition(b"\x00")
                chunks[key.decode("utf-8")] = (zlib.decompress(text) if compressed else text).decode("utf-8")
        except (ValueError, zlib.error, IndexError):
            continue
        if kind == b"IEND":
            break
    return chunks


def load_prompt(path: str):
    """The ``prompt`` graph from a PNG (its ``prompt`` chunk) or a JSON file, or None."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data.startswith(PNG_SIGNATURE):
        raw = png_text_chunks(data).get("prompt")
        if raw is None:
            return None
        data = raw.encode("utf-8")
    graph = json.loads(data)
    if isinstance(graph, dict) and isinstance(graph.get("prompt"), dict) and "class_type" not in graph["prompt"]:
        graph = graph["prompt"]  # {"prompt": {...}} as ComfyUI's queue endpoint takes it
    return graph if isinstance(graph, dict) else None


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    trace = "--trace" in args
    paths = [a for a in args if a != "--trace"]
    if len(paths) != 1 or paths[0].startswith("-"):
        print("usage: python3 graphwalk.py <image.png | prompt.json> [--trace]", file=sys.stderr)
        return 2
    try:
        prompt = load_prompt(paths[0])
    except (OSError, ValueError) as exc:
        print(f"cannot read {paths[0]}: {exc}", file=sys.stderr)
        return 2
    if prompt is None:
        print("no ComfyUI 'prompt' graph found", file=sys.stderr)
        return 1
    info, lines = trace_from_prompt(prompt)
    if info is None:
        print("no known sampler node in the graph", file=sys.stderr)
        return 1
    print("\n".join(lines) if trace else format_a1111(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
