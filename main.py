#!/usr/bin/env python3
"""
ComfyUI Graph-Walk parser plugin for sd-webui-infinite-image-browsing (IIB).

Replaces IIB's built-in ComfyUIParser, which only reads the raw KSampler inputs and
so returns wrong values on non-trivial workflows. This plugin walks the ComfyUI
``prompt`` graph backwards from the sampler, resolving the real values through
Reroute / LoRA-loader / conditioning / switch chains, then builds an A1111-style
``parameters`` string and hands it to IIB's ``parse_generation_parameters``.

``scripts/iib/parsers/index.py`` prepends plugin instances to the parser list, so
this runs before the built-in parser and transparently supersedes it. See README.md
for behaviour, supported nodes and install steps.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Mikolas Bingemer
"""

import json
import re
from pathlib import Path

from scripts.iib.parsers.model import ImageGenerationInfo, ImageGenerationParams
from scripts.iib.tool import (
    parse_generation_parameters,
    find,
)

# ``is_img_created_by_comfyui`` lives in ``scripts.iib.tool`` on older IIB
# (≤ v1.8.0) but was moved to ``scripts.iib.parsers.comfyui`` in newer versions.
# Import defensively so the plugin loads across both — otherwise the whole
# import fails and IIB silently falls back to its built-in ComfyUI parser.
try:
    from scripts.iib.tool import is_img_created_by_comfyui
except ImportError:
    from scripts.iib.parsers.comfyui import is_img_created_by_comfyui

# Private IIB helper used only for the non-standard EXIF fallback. It carries a
# leading underscore, so it may be renamed or removed across IIB versions — import
# it defensively so its absence degrades one fallback path instead of disabling the
# whole plugin.
try:
    from scripts.iib.tool import _extract_usercomment_from_raw_exif
except ImportError:
    _extract_usercomment_from_raw_exif = None

# ---------------------------------------------------------------------------
# Known node classes
# ---------------------------------------------------------------------------

SAMPLER_CLASSES = {
    "KSampler",
    "KSamplerAdvanced",
    # efficiency-nodes
    "KSampler (Efficient)",
    "KSamplerAdvanced (Efficient)",
    "Tiled KSampler",
    "BNK_Unsampler",
}

# Nodes that pass through conditioning signals unchanged (no text of their own)
COND_PASSTHROUGH = {
    "Reroute",
    "ConditioningSetTimestepRange",
    "ConditioningSetArea",
    "ConditioningSetAreaPercentage",
    "ConditioningSetMask",
    "unCLIPConditioning",
}

TEXT_ENCODE_CLASSES = {
    "CLIPTextEncode",
    "CLIPTextEncodeSDXL",
    "CLIPTextEncodeSDXLRefiner",
    "smZ CLIPTextEncode",
    "BNK_CLIPTextEncodeAdvanced",
}

# Nodes that merge multiple conditioning inputs into one
COND_COMBINE = {
    "ConditioningCombine",
    "ConditioningConcat",
    "ConditioningAverage",
}

# Checkpoint loader classes (expose the model name)
CHECKPOINT_CLASSES = {
    "CheckpointLoaderSimple",
    "CheckpointLoader",
    "unCLIPCheckpointLoader",
    "Checkpoint Loader",
}

# Reference list of nodes that forward the model output on a ``model`` edge.
# ``resolve_model_name`` no longer gates on this set — it follows any ``model`` edge
# generically so unknown patchers are traversed too — but the list is kept as
# documentation of the commonly-seen model-passthrough classes.
MODEL_PASSTHROUGH = {
    "LoraLoader",
    "LoraLoaderModelOnly",
    "Lora Loader (LoraManager)",
    "Power Lora Loader (rgthree)",
    "Lora Loader Stack (rgthree)",
    "ModelMergeSimple",
    "ModelMergeAdd",
    "ModelMergeBlocks",
    "CLIPSetLastLayer",
    "ModelSamplingDiscrete",
    "ModelSamplingContinuousEDM",
    "Mahiro",
    "CFGZeroStar",
    "Epsilon Scaling",
    "IPAdapterFaceID",
    "IPAdapterAdvanced",
    "IPAdapterStyleComposition",
    "AttentionCouplePPM",
    "FreeU",
    "FreeU_V2",
    "PerturbedAttentionGuidance",
    "SelfAttentionGuidance",
    "RescaleCFG",
    "DifferentialDiffusion",
    "Reroute",
}

# Latent image nodes (source of width/height resolution)
LATENT_SIZE_CLASSES = {
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyHunyuanLatentVideo",
}

# A "W×H" / "WxH" size inside a label, e.g. "Portrait 2:3 · 1152×1728 · 1.9 MP".
SIZE_IN_LABEL = re.compile(r"(\d+)\s*[×x]\s*(\d+)")

# Display / debug nodes that echo an upstream value into a cached widget and pass
# it through unchanged. When resolving through them we read the cached widget (the
# last-run value, reliable for reproduced images) or follow the passthrough input.
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

# Nodes that output one of their own text inputs unchanged, mapped to that input's
# name. For inputs not covered by the generic text keys in resolve_text's fallback
# ("value", "text", …) whose name can't be added there because it means something
# else on other nodes (TextGenerate's 'prompt' is the LLM's input, not its output).
TEXT_PASSTHROUGH = {
    "LLMPromptEnhancer": "prompt",
}

# ---------------------------------------------------------------------------
# Graph-walk helpers
# ---------------------------------------------------------------------------


def _node(prompt: dict, node_id):
    return prompt.get(str(node_id))


def _is_switch(class_type: str) -> bool:
    """True for the various ComfyUI *Switch nodes (ImpactSwitch, Any Switch, …)."""
    return "Switch" in class_type


def _switch_selected(prompt: dict, class_type: str, inputs: dict):
    """
    For a Switch node, return the graph-edge reference of the selected input
    (``input{select}``, 1-based). Falls back to the first connected ``input*``
    when ``select`` can't be resolved. Returns None for non-switch nodes.
    """
    if not _is_switch(class_type):
        return None
    select = inputs.get("select", 1)
    if isinstance(select, list):
        select = get_scalar(prompt, inputs, "select") or 1
    try:
        select = int(select)
    except (TypeError, ValueError):
        select = 1
    val = inputs.get(f"input{select}")
    if isinstance(val, list):
        return val
    # Fallback: first connected input slot when 'select' points at an empty one.
    for i in range(1, 32):
        val = inputs.get(f"input{i}")
        if isinstance(val, list):
            return val
    return None


def resolve_text(prompt: dict, ref, _seen: set | None = None) -> str:
    """
    Resolve a reference or raw value to a string.
    ref is either a plain string/int/float or a graph-edge reference [node_id, out_idx].
    """
    if _seen is None:
        _seen = set()

    if isinstance(ref, str):
        return ref
    if isinstance(ref, (int, float)):
        return str(ref)
    if not isinstance(ref, list):
        return ""

    node_id = str(ref[0])
    if node_id in _seen:
        return ""
    _seen = _seen | {node_id}

    node = _node(prompt, node_id)
    if not node:
        return ""

    class_type = node.get("class_type", "")
    inputs = node.get("inputs", {})

    if class_type == "Reroute":
        return resolve_text(prompt, inputs.get("*"), _seen)

    # Switch nodes (ImpactSwitch, Any Switch, …): follow the selected branch.
    sel = _switch_selected(prompt, class_type, inputs)
    if sel is not None:
        return resolve_text(prompt, sel, _seen)

    # Display / debug passthrough (easy showAnything, ShowText, …): the resolved
    # value is cached in a text widget; fall back to the passthrough input edge.
    if class_type in SHOW_PASSTHROUGH:
        for key in ("text", "value", "string"):
            val = inputs.get(key)
            if isinstance(val, str) and val:
                return val
        for key in ("anything", "source", "input", "value", "text"):
            val = inputs.get(key)
            if isinstance(val, list):
                return resolve_text(prompt, val, _seen)

    # ImpactWildcardProcessor (ComfyUI-Impact-Pack):
    # 'populated_text' holds the already-resolved wildcard text (when mode='reproduce')
    if class_type == "ImpactWildcardProcessor":
        pt = inputs.get("populated_text", "")
        if isinstance(pt, list):
            return resolve_text(prompt, pt, _seen)
        return str(pt) if pt else ""

    # RegexReplace: follow the string input, then apply the regex substitution
    if class_type == "RegexReplace":
        text = resolve_text(prompt, inputs.get("string"), _seen)
        if text:
            pattern = inputs.get("regex_pattern", "")
            replace = inputs.get("replace", "")
            count = int(inputs.get("count") or 0)
            flags = 0
            if inputs.get("case_insensitive"):
                flags |= re.IGNORECASE
            if inputs.get("multiline"):
                flags |= re.MULTILINE
            if inputs.get("dotall"):
                flags |= re.DOTALL
            try:
                text = re.sub(pattern, replace, text, count=count, flags=flags)
            except re.error:
                pass
        return text

    # StringFunction|pysssss and similar string-concatenation nodes
    if class_type in ("StringFunction|pysssss", "StringFunction"):
        action = inputs.get("action", "append")
        parts = []
        for key in ("text_a", "text_b", "text_c"):
            val = inputs.get(key)
            if val is None:
                continue
            t = resolve_text(prompt, val, _seen)
            if t:
                parts.append(t)
        sep = ", " if action in ("append",) else ""
        return sep.join(parts)

    # StringConcatenate (comfy-core): joins string_a and string_b with a delimiter widget
    if class_type == "StringConcatenate":
        delimiter = inputs.get("delimiter", " ")
        if isinstance(delimiter, list):
            delimiter = resolve_text(prompt, delimiter, _seen) or " "
        parts = []
        for key in ("string_a", "string_b"):
            val = inputs.get(key)
            if val is None:
                continue
            t = resolve_text(prompt, val, _seen)
            if t:
                parts.append(t)
        return delimiter.join(parts)

    # Nodes that output one of their text inputs unchanged (see TEXT_PASSTHROUGH).
    if class_type in TEXT_PASSTHROUGH:
        return resolve_text(prompt, inputs.get(TEXT_PASSTHROUGH[class_type]), _seen)

    # Primitive / ShowText / Note and others — read direct scalar text values
    for key in ("value", "text", "string", "populated_text", "trigger_words"):
        val = inputs.get(key)
        if val is not None and not isinstance(val, list):
            return str(val)
        if isinstance(val, list):
            return resolve_text(prompt, val, _seen)

    return ""


def get_conditioning_text(prompt: dict, cond_ref, _seen: set | None = None) -> str:
    """
    Walk a conditioning edge backwards until the originating text node is reached.
    cond_ref = [node_id, out_idx]
    """
    if not isinstance(cond_ref, list):
        return ""
    if _seen is None:
        _seen = set()

    node_id = str(cond_ref[0])
    if node_id in _seen:
        return ""
    _seen = _seen | {node_id}

    node = _node(prompt, node_id)
    if not node:
        return ""

    class_type = node.get("class_type", "")
    inputs = node.get("inputs", {})

    # Text-encode node reached — extract the text
    if class_type in TEXT_ENCODE_CLASSES:
        # CLIPTextEncodeSDXL exposes both text_g (global) and text_l (local)
        parts = []
        for key in ("text_g", "text_l", "text"):
            val = inputs.get(key)
            if val is None:
                continue
            t = resolve_text(prompt, val, _seen)
            if t and t not in parts:
                parts.append(t)
        return "\n".join(parts) if parts else ""

    # Switch nodes routing conditioning — follow the selected branch.
    sel = _switch_selected(prompt, class_type, inputs)
    if sel is not None:
        return get_conditioning_text(prompt, sel, _seen)

    # Pure passthrough (Reroute, ConditioningSetX…) — follow the conditioning input
    if class_type in COND_PASSTHROUGH:
        for key in ("*", "conditioning"):
            val = inputs.get(key)
            if isinstance(val, list):
                return get_conditioning_text(prompt, val, _seen)
        return ""

    # Combine nodes — merge text from both conditioning inputs
    if class_type in COND_COMBINE:
        parts = []
        for key in ("conditioning_1", "conditioning_2", "conditioning_to", "conditioning_from"):
            val = inputs.get(key)
            if isinstance(val, list):
                t = get_conditioning_text(prompt, val, _seen)
                if t:
                    parts.append(t)
        return ", ".join(parts)

    # ControlNetApply nodes pass conditioning through slot-matched inputs:
    # output slot 0 = positive conditioning (input key "positive"),
    # output slot 1 = negative conditioning (input key "negative").
    if class_type in ("ControlNetApplyAdvanced", "ControlNetApply"):
        out_slot = cond_ref[1] if len(cond_ref) > 1 else 0
        follow_key = "negative" if out_slot == 1 else "positive"
        val = inputs.get(follow_key)
        if isinstance(val, list):
            return get_conditioning_text(prompt, val, _seen)
        return ""

    # Unknown node type with a conditioning input — follow it anyway
    for key in ("conditioning", "conditioning_1"):
        val = inputs.get(key)
        if isinstance(val, list):
            return get_conditioning_text(prompt, val, _seen)

    return ""


def resolve_model_name(prompt: dict, model_ref, _seen: set | None = None) -> str:
    """
    Walk the model edge backwards through LoRA loaders and similar passthrough nodes
    until the checkpoint loader is reached; return its ckpt_name.
    """
    if not isinstance(model_ref, list):
        return ""
    if _seen is None:
        _seen = set()

    node_id = str(model_ref[0])
    if node_id in _seen:
        return ""
    _seen = _seen | {node_id}

    node = _node(prompt, node_id)
    if not node:
        return ""

    class_type = node.get("class_type", "")
    inputs = node.get("inputs", {})

    if class_type in CHECKPOINT_CLASSES:
        return inputs.get("ckpt_name") or inputs.get("unet_name") or ""
    # Some loaders expose the model under a different widget without being a
    # dedicated checkpoint class — honour those names when present.
    for key in ("ckpt_name", "unet_name", "model_name"):
        val = inputs.get(key)
        if isinstance(val, str) and val:
            return val

    # Otherwise treat any node carrying a model edge as a passthrough (LoRA
    # loaders, model patchers, CLIP*Pip, IPAdapter, samplers-config, …) and keep
    # walking. This is deliberately generic: rather than whitelisting every
    # model-patching node, follow the ``model`` input back to the loader.
    sel = _switch_selected(prompt, class_type, inputs)
    if sel is not None:
        return resolve_model_name(prompt, sel, _seen)

    sub = inputs.get("*") if class_type == "Reroute" else inputs.get("model")
    if isinstance(sub, list):
        return resolve_model_name(prompt, sub, _seen)

    return ""


# Matches <lora:name:weight> and <lora:name:model_weight:clip_weight> (same
# pattern as LoraManager's parse_lora_syntax); only the model weight is kept.
_LORA_TAG_RE = re.compile(r"<lora:([^:>]+):([^:>]+)(?::[^:>]+)?>", re.IGNORECASE)

# Weight-file extensions stripped from LoRA names. Not Path.stem: LoraManager
# names carry no extension, so a dot inside the name ("v1.5") must survive.
_LORA_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")

LM_LOADER = "Lora Loader (LoraManager)"
LM_TEXT_LOADER = "LoRA Text Loader (LoraManager)"
LM_STACKER = "Lora Stacker (LoraManager)"


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


def _lm_stack_loras(prompt: dict, stack_ref, _seen: set | None = None):
    """Follow a LoraManager ``lora_stack`` link through chained Stacker nodes."""
    if not isinstance(stack_ref, list):
        return []
    if _seen is None:
        _seen = set()
    node_id = str(stack_ref[0])
    if node_id in _seen:
        return []
    node = _node(prompt, node_id)
    if not node or node.get("class_type") != LM_STACKER:
        return []
    inputs = node.get("inputs", {})
    # The stacker puts the incoming stack first, then its own active entries.
    out = _lm_stack_loras(prompt, inputs.get("lora_stack"), _seen | {node_id})
    out.extend(_lm_widget_loras(inputs.get("loras")) or [])
    return out


def _node_loras(prompt: dict, class_type: str, inputs: dict):
    """
    Extract (lora_name, weight) pairs configured directly on a single node.
    Handles the common ComfyUI LoRA-loader variants.
    """
    out = []

    # LoraManager loader: applies its incoming lora_stack, then the active
    # entries of its ``loras`` widget. The ``text`` widget is only an editor
    # mirror that still lists deactivated LoRAs, so it is read only when the
    # structured list is missing.
    if class_type == LM_LOADER:
        out.extend(_lm_stack_loras(prompt, inputs.get("lora_stack")))
        widget = _lm_widget_loras(inputs.get("loras"))
        if widget is None:
            text = inputs.get("text")
            widget = _lora_tags_in_text(text) if isinstance(text, str) else []
        out.extend(widget)
        return out

    # LoraManager text loader: lora_stack, then the <lora:…> tags of lora_syntax.
    if class_type == LM_TEXT_LOADER:
        out.extend(_lm_stack_loras(prompt, inputs.get("lora_stack")))
        syntax = inputs.get("lora_syntax")
        if isinstance(syntax, list):
            syntax = resolve_text(prompt, syntax)
        if isinstance(syntax, str):
            out.extend(_lora_tags_in_text(syntax))
        return out

    # rgthree Power Lora Loader: one entry per ``lora_N`` key, each a dict
    # {"on": bool, "lora": name, "strength": weight}.
    if class_type == "Power Lora Loader (rgthree)":
        for key, val in inputs.items():
            if not re.fullmatch(r"lora_\d+", key) or not isinstance(val, dict):
                continue
            name = val.get("lora")
            if val.get("on", True) and isinstance(name, str) and name.lower() != "none":
                out.append((name, val.get("strength", 1.0)))
        return out

    # Core LoraLoader / LoraLoaderModelOnly.
    if class_type in ("LoraLoader", "LoraLoaderModelOnly"):
        name = get_scalar(prompt, inputs, "lora_name")
        if isinstance(name, str) and name.lower() != "none":
            weight = inputs.get("strength_model")
            if weight is None:
                weight = inputs.get("strength", 1.0)
            out.append((name, weight))

    # rgthree Lora Loader Stack: lora_1..N / strength_1..N, "None" = empty slot.
    if class_type == "Lora Loader Stack (rgthree)":
        i = 1
        while f"lora_{i}" in inputs:
            name = inputs.get(f"lora_{i}")
            if isinstance(name, str) and name.lower() != "none":
                out.append((name, inputs.get(f"strength_{i}", 1.0)))
            i += 1

    return out


def collect_loras(prompt: dict, model_ref, _seen: set | None = None):
    """
    Walk the model edge backwards (like resolve_model_name) and collect every
    LoRA applied along the chain as (name, weight) pairs, de-duplicated by the
    filename stem (first weight wins).
    """
    if not isinstance(model_ref, list):
        return []
    if _seen is None:
        _seen = set()

    node_id = str(model_ref[0])
    if node_id in _seen:
        return []
    _seen = _seen | {node_id}

    node = _node(prompt, node_id)
    if not node:
        return []

    class_type = node.get("class_type", "")
    inputs = node.get("inputs", {})

    loras = list(_node_loras(prompt, class_type, inputs))

    if class_type == "Reroute":
        sub = inputs.get("*")
    else:
        sub = _switch_selected(prompt, class_type, inputs) or inputs.get("model")
    if isinstance(sub, list):
        loras.extend(collect_loras(prompt, sub, _seen))

    # De-duplicate by stem, preserving first occurrence.
    result = []
    seen_stems = set()
    for name, weight in loras:
        stem = _lora_stem(name).lower()
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        result.append((name, weight))
    return result


def get_scalar(prompt: dict, inputs: dict, key: str, _seen: set | None = None):
    """
    Read a scalar input value, following graph-edge references.

    ``key`` is the *name of the input we are resolving* (e.g. "steps", "cfg",
    "sampler_name"). It is carried through the walk so that multi-output source
    nodes — which key their widgets by name, not by output slot — can be resolved
    by matching that name. This handles aggregator nodes such as
    ``Input Parameters (Image Saver)`` where ``KSampler.steps`` points at output
    slot 1 but the value lives in a widget literally called ``steps``.
    """
    return _resolve_scalar(prompt, inputs.get(key), key, _seen)


def _resolve_scalar(prompt: dict, val, key: str, _seen: set | None = None):
    if val is None:
        return None
    if not isinstance(val, list):
        return val

    if _seen is None:
        _seen = set()
    node_id = str(val[0])
    if node_id in _seen:
        return None
    _seen = _seen | {node_id}

    node = _node(prompt, node_id)
    if not node:
        return None

    class_type = node.get("class_type", "")
    sub = node.get("inputs", {})

    # Switch — follow the selected branch, keeping the same requested key.
    sel = _switch_selected(prompt, class_type, sub)
    if sel is not None:
        return _resolve_scalar(prompt, sel, key, _seen)

    # A widget named exactly like the input we're after (aggregator/pipe nodes).
    if key in sub:
        return _resolve_scalar(prompt, sub[key], key, _seen)

    # Display / debug passthrough — read the cached widget or follow the input.
    if class_type in SHOW_PASSTHROUGH:
        for k in ("text", "value", "int", "float", "string"):
            v = sub.get(k)
            if v is not None and not isinstance(v, list):
                return v
        for k in ("anything", "source", "input", "value", "text"):
            v = sub.get(k)
            if isinstance(v, list):
                return _resolve_scalar(prompt, v, key, _seen)

    # Size pickers without width/height widgets: take the first "W×H" from a string
    # widget (preset label such as "Portrait 2:3 · 1152×1728 · 1.9 MP"); otherwise a
    # widget whose name contains the requested key (custom_width / custom_height).
    # Only literal strings are scanned, linked inputs are not followed.
    if key in ("width", "height"):
        for v in sub.values():
            size = SIZE_IN_LABEL.search(v) if isinstance(v, str) else None
            if size:
                return int(size.group(1 if key == "width" else 2))
        for k, v in sub.items():
            if key in k:
                r = _resolve_scalar(prompt, v, k, _seen)
                if r is not None:
                    return r

    # Generic scalar source (Primitive, Seed, easy int/float, …).
    for k in ("value", "int", "float", "seed", "number", "result"):
        v = sub.get(k)
        if v is not None and not isinstance(v, list):
            return v
        if isinstance(v, list):
            r = _resolve_scalar(prompt, v, key, _seen)
            if r is not None:
                return r
    return None


def resolve_latent_size(prompt: dict, latent_ref, _seen: set | None = None):
    """Return (width, height) if the latent originates from an EmptyLatentImage node, otherwise (None, None)."""
    if not isinstance(latent_ref, list):
        return None, None
    if _seen is None:
        _seen = set()

    node_id = str(latent_ref[0])
    if node_id in _seen:
        return None, None
    _seen = _seen | {node_id}

    node = _node(prompt, node_id)
    if not node:
        return None, None

    class_type = node.get("class_type", "")
    inputs = node.get("inputs", {})

    if class_type in LATENT_SIZE_CLASSES:
        # width/height may be direct values or references to upstream scalar (Primitive) nodes
        w = get_scalar(prompt, inputs, "width")
        h = get_scalar(prompt, inputs, "height")
        return w, h

    # Reroute
    if class_type == "Reroute":
        return resolve_latent_size(prompt, inputs.get("*"), _seen)

    # ImpactSwitch / Any Switch: follow the selected branch.
    sel = _switch_selected(prompt, class_type, inputs)
    if sel is not None:
        return resolve_latent_size(prompt, sel, _seen)

    # Latent-passthrough nodes (upscale, VAE round-trips, samplers, …) keep the
    # resolution — follow the first latent-carrying edge back to its source.
    for key in ("latent", "latent_image", "samples", "LATENT"):
        val = inputs.get(key)
        if isinstance(val, list):
            return resolve_latent_size(prompt, val, _seen)

    return None, None


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_from_prompt(prompt: dict) -> dict | None:
    """
    Find the primary sampler node in the ComfyUI graph and extract all relevant
    generation parameters. Returns None when no sampler node is found.
    """
    sampler_nodes = [
        (nid, node)
        for nid, node in prompt.items()
        if isinstance(node, dict) and node.get("class_type", "") in SAMPLER_CLASSES
    ]

    if not sampler_nodes:
        return None

    # Prefer KSampler / KSamplerAdvanced over specialised variants
    primary = next(
        ((nid, n) for nid, n in sampler_nodes if n.get("class_type") in ("KSampler", "KSamplerAdvanced")),
        sampler_nodes[0],
    )
    _, node = primary
    class_type = node.get("class_type", "")
    inputs = node.get("inputs", {})

    positive = get_conditioning_text(prompt, inputs.get("positive"))
    negative = get_conditioning_text(prompt, inputs.get("negative"))

    seed_key = "noise_seed" if class_type == "KSamplerAdvanced" else "seed"
    seed = get_scalar(prompt, inputs, seed_key)
    steps = get_scalar(prompt, inputs, "steps")
    cfg = get_scalar(prompt, inputs, "cfg")
    sampler_name = get_scalar(prompt, inputs, "sampler_name")
    scheduler = get_scalar(prompt, inputs, "scheduler")

    model_name = resolve_model_name(prompt, inputs.get("model"))
    width, height = resolve_latent_size(prompt, inputs.get("latent_image"))
    loras = collect_loras(prompt, inputs.get("model"))

    return {
        "positive": positive,
        "negative": negative,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "model_name": model_name,
        "width": width,
        "height": height,
        "loras": loras,
    }


# ---------------------------------------------------------------------------
# A1111 output formatting
# ---------------------------------------------------------------------------


def _lora_tags(loras) -> str:
    """Render collected LoRAs as A1111 ``<lora:name:weight>`` tags. IIB (and its
    webui "Resources" panel) reads LoRAs from these prompt tags, so emitting them
    surfaces the graph's LoRAs the same way A1111 metadata would."""
    tags = []
    for name, weight in loras or []:
        try:
            w = f"{float(weight):g}"
        except (TypeError, ValueError):
            w = "1"
        tags.append(f"<lora:{_lora_stem(name)}:{w}>")
    return " ".join(tags)


def format_a1111(info: dict) -> str:
    """Format extracted generation fields as an A1111-style parameters string."""
    lines = []

    positive = (info.get("positive") or "").strip()
    lora_tags = _lora_tags(info.get("loras"))
    if lora_tags:
        positive = f"{positive} {lora_tags}".strip()
    if positive:
        lines.append(positive)

    neg = (info.get("negative") or "").strip()
    if neg:
        lines.append(f"Negative prompt: {neg}")

    parts = []
    if info.get("steps") is not None:
        parts.append(f"Steps: {info['steps']}")
    if info.get("sampler_name"):
        samp = info["sampler_name"]
        if info.get("scheduler"):
            samp = f"{samp} {info['scheduler']}"
        parts.append(f"Sampler: {samp}")
    if info.get("cfg") is not None:
        parts.append(f"CFG scale: {info['cfg']}")
    if info.get("seed") is not None:
        parts.append(f"Seed: {info['seed']}")
    if info.get("width") and info.get("height"):
        parts.append(f"Size: {info['width']}x{info['height']}")
    if info.get("model_name"):
        parts.append(f"Model: {Path(info['model_name']).stem}")

    if parts:
        lines.append(", ".join(parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading the ComfyUI prompt chunk across image formats
# ---------------------------------------------------------------------------


def _get_prompt_dict(img):
    """Read the ComfyUI prompt chunk from PNG, WEBP, or JPEG; return the parsed dict or None."""
    raw = None
    if img.format == "PNG":
        raw = img.info.get("prompt")
    else:  # WEBP / JPEG: prompt lives in the EXIF block
        exif = img.info.get("exif")
        if exif:
            split = [x.decode("utf-8", "ignore") for x in exif.split(b"\x00")]
            s = find(split, lambda x: x.lower().startswith("prompt:"))
            if s:
                raw = s.split(":", 1)[1]
            elif _extract_usercomment_from_raw_exif is not None:
                # Fallback: non-standard EXIF (UserComment in IFD0)
                rawc = _extract_usercomment_from_raw_exif(exif)
                if rawc:
                    dec = (
                        rawc.decode("utf-16", "ignore").strip("\x00")
                        or rawc.decode("utf-8", "ignore").strip("\x00")
                    )
                    p = find([dec], lambda x: x.lower().startswith("prompt:"))
                    raw = p.split(":", 1)[1] if p else None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


class Main:
    name = "ComfyUI Graph-Walk"
    source_identifier = "ComfyUI"
    extra_convert_to_tag_meta_keys = []

    def test(self, img, file_path) -> bool:
        try:
            return bool(is_img_created_by_comfyui(img))
        except Exception:
            return False

    def parse(self, img, file_path) -> ImageGenerationInfo:
        width, height = img.size
        try:
            prompt = _get_prompt_dict(img)
            info = extract_from_prompt(prompt) if prompt else None
            if not info:
                raise ValueError("no sampler node found")

            a1111 = format_a1111(info)
            # "Source Identifier" must appear on the metadata line to match ComfyUIParser behaviour.
            a1111 = (a1111 + ", Source Identifier: ComfyUI") if a1111 else "Source Identifier: ComfyUI"

            params = parse_generation_parameters(a1111)
            return ImageGenerationInfo(
                a1111,
                ImageGenerationParams(
                    meta=params["meta"] | {"final_width": width, "final_height": height},
                    pos_prompt=params["pos_prompt"],
                    extra=params,
                ),
            )
        except Exception:
            # Mirror built-in parser behaviour: on error, still return image dimensions and never re-raise.
            return ImageGenerationInfo(
                params=ImageGenerationParams(
                    meta={"final_width": width, "final_height": height}
                )
            )
