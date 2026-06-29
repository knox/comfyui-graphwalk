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
    is_img_created_by_comfyui,
    parse_generation_parameters,
    find,
)

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

# Nodes that forward the model output — traverse these back to the checkpoint loader
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

# ---------------------------------------------------------------------------
# Graph-walk helpers
# ---------------------------------------------------------------------------


def _node(prompt: dict, node_id):
    return prompt.get(str(node_id))


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

    if class_type in MODEL_PASSTHROUGH:
        sub = inputs.get("model")
        if isinstance(sub, list):
            return resolve_model_name(prompt, sub, _seen)

    return ""


def get_scalar(prompt: dict, inputs: dict, key: str):
    """
    Read a scalar input value, following graph-edge references (e.g. from a Primitive node).
    """
    val = inputs.get(key)
    if val is None:
        return None
    if isinstance(val, list):
        node = _node(prompt, val[0])
        if node:
            sub = node.get("inputs", {})
            for k in ("value", "int", "float", "seed", "number"):
                v = sub.get(k)
                if v is not None and not isinstance(v, list):
                    return v
        return None
    return val


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

    # ImpactSwitch / Any Switch: 1-based 'select' index determines which input to follow
    if "Switch" in class_type or class_type in ("ImpactSwitch", "ImpactInversedSwitch"):
        select = inputs.get("select", 1)
        if isinstance(select, list):
            select = get_scalar(prompt, inputs, "select") or 1
        try:
            select = int(select)
        except (TypeError, ValueError):
            select = 1
        val = inputs.get(f"input{select}")
        if isinstance(val, list):
            return resolve_latent_size(prompt, val, _seen)
        # Fallback: try the first available input* field when 'select' cannot be resolved
        for i in range(1, 5):
            val = inputs.get(f"input{i}")
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
    }


# ---------------------------------------------------------------------------
# A1111 output formatting
# ---------------------------------------------------------------------------


def format_a1111(info: dict) -> str:
    """Format extracted generation fields as an A1111-style parameters string."""
    lines = []

    if info.get("positive"):
        lines.append(info["positive"].strip())

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
