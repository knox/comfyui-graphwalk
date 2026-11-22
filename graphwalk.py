#!/usr/bin/env python3
"""
ComfyUI Graph-Walk core: generation parameters from a ComfyUI ``prompt`` graph.

Walks the ComfyUI API ``prompt`` graph backwards from the sampler, resolving the
real values through Reroute / LoRA-loader / conditioning / switch chains, and
formats them as an A1111-style ``parameters`` string.

Standard library only, no IIB or PIL: the caller reads the ``prompt`` chunk from
the image and passes the parsed dict. The IIB plugin (``main.py``) is a thin
adapter around this module. Public interface:

- ``extract_from_prompt(prompt: dict) -> dict | None``
- ``format_a1111(info: dict) -> str``

See README.md for behaviour and supported nodes.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Mikolas Bingemer
"""

import re
from pathlib import PurePosixPath

__all__ = [
    "extract_from_prompt",
    "format_a1111",
    "a1111_prompt_text",
    "a1111_params",
    "lora_entries",
    "model_basename",
]

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
    node = prompt.get(str(node_id))
    return node if isinstance(node, dict) else None


def _inputs(node: dict) -> dict:
    inputs = node.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def _is_ref(val) -> bool:
    """True for a graph-edge reference ``[node_id, output_index]``."""
    return isinstance(val, list) and len(val) >= 1 and isinstance(val[0], (str, int))


def _is_switch(class_type: str) -> bool:
    """True for the various ComfyUI *Switch nodes (ImpactSwitch, Any Switch, …)."""
    return "Switch" in class_type


# comfy-core If/Else switches: inputs ``switch`` (bool), ``on_false``, ``on_true``.
BOOL_SWITCHES = {"ComfySwitchNode", "ComfySoftSwitchNode"}
# rgthree: outputs the first non-empty of its ``any_01``, ``any_02``, … inputs.
ANY_SWITCH = "Any Switch (rgthree)"


def _truthy(val) -> bool:
    if isinstance(val, str):
        return val.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(val)


def _switch_selected(prompt: dict, class_type: str, inputs: dict):
    """
    For a Switch node, return the graph-edge reference of the selected input.
    Falls back to the first connected input when the selection can't be resolved.
    Returns None for non-switch nodes (and when nothing is selected).

    - comfy-core ``ComfySwitchNode`` / ``ComfySoftSwitchNode``: ``on_true`` /
      ``on_false`` by the ``switch`` boolean. The soft variant uses the connected
      branch when only one is connected, the hard one outputs nothing.
    - rgthree ``Any Switch``: the first connected ``any_*`` input.
    - Everything else with "Switch" in its name (Impact, …): ``input{select}``.
    """
    if class_type in BOOL_SWITCHES:
        flag = _truthy(get_scalar(prompt, inputs, "switch"))
        chosen = inputs.get("on_true" if flag else "on_false")
        if _is_ref(chosen):
            return chosen
        if class_type == "ComfySoftSwitchNode":
            other = inputs.get("on_false" if flag else "on_true")
            if _is_ref(other):
                return other
        return None
    if class_type == ANY_SWITCH:
        # By slot number, not by the key order of the stored prompt.
        slots = sorted(
            (int(key[4:]), val) for key, val in inputs.items() if re.fullmatch(r"any_\d+", key) and _is_ref(val)
        )
        return slots[0][1] if slots else None
    if not _is_switch(class_type):
        return None
    select = inputs.get("select", 1)
    if _is_ref(select):
        select = get_scalar(prompt, inputs, "select") or 1
    try:
        select = int(select)
    except (TypeError, ValueError):
        select = 1
    val = inputs.get(f"input{select}")
    if _is_ref(val):
        return val
    # Fallback: first connected input slot when 'select' points at an empty one.
    for i in range(1, 32):
        val = inputs.get(f"input{i}")
        if _is_ref(val):
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
    if not _is_ref(ref):
        return ""

    node_id = str(ref[0])
    if node_id in _seen:
        return ""
    _seen = _seen | {node_id}

    node = _node(prompt, node_id)
    if not node:
        return ""

    class_type = node.get("class_type", "")
    inputs = _inputs(node)

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
            if _is_ref(val):
                return resolve_text(prompt, val, _seen)

    # ImpactWildcardProcessor (ComfyUI-Impact-Pack):
    # 'populated_text' holds the already-resolved wildcard text (when mode='reproduce')
    if class_type == "ImpactWildcardProcessor":
        pt = inputs.get("populated_text", "")
        if _is_ref(pt):
            return resolve_text(prompt, pt, _seen)
        return str(pt) if pt else ""

    # RegexReplace: follow the string input, then apply the regex substitution.
    # The pattern, replacement, count and flags may be wired from other nodes.
    if class_type == "RegexReplace":
        text = resolve_text(prompt, inputs.get("string"), _seen)
        if text:
            pattern = resolve_text(prompt, inputs.get("regex_pattern", ""), _seen)
            replace = resolve_text(prompt, inputs.get("replace", ""), _seen)
            try:
                count = int(get_scalar(prompt, inputs, "count") or 0)
            except (TypeError, ValueError):
                count = 0
            flags = 0
            if _truthy(get_scalar(prompt, inputs, "case_insensitive")):
                flags |= re.IGNORECASE
            if _truthy(get_scalar(prompt, inputs, "multiline")):
                flags |= re.MULTILINE
            if _truthy(get_scalar(prompt, inputs, "dotall")):
                flags |= re.DOTALL
            try:
                text = re.sub(pattern, replace, text, count=count, flags=flags)
            except (re.error, ValueError):
                pass
        return text

    # StringFunction|pysssss: same logic as the node's ``exec``.
    if class_type in ("StringFunction|pysssss", "StringFunction"):
        return _string_function(prompt, inputs, _seen)

    # StringConcatenate (comfy-core): ``delimiter.join((string_a, string_b))``;
    # the delimiter widget defaults to "" and empty parts are kept.
    if class_type == "StringConcatenate":
        delimiter = inputs.get("delimiter", "")
        if _is_ref(delimiter):
            delimiter = resolve_text(prompt, delimiter, _seen)
        part_a = resolve_text(prompt, inputs.get("string_a"), _seen)
        part_b = resolve_text(prompt, inputs.get("string_b"), _seen)
        return str(delimiter if delimiter is not None else "").join((part_a, part_b))

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


def _string_function(prompt: dict, inputs: dict, _seen: set) -> str:
    """``StringFunction|pysssss``: append or replace, then optionally tidy tags."""
    action = inputs.get("action", "append")
    tidy = inputs.get("tidy_tags", "yes") == "yes"
    text_a, text_b, text_c = (resolve_text(prompt, inputs.get(k), _seen) for k in ("text_a", "text_b", "text_c"))
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


def get_conditioning_text(prompt: dict, cond_ref, _seen: set | None = None) -> str:
    """
    Walk a conditioning edge backwards until the originating text node is reached.
    cond_ref = [node_id, out_idx]
    """
    if not _is_ref(cond_ref):
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
    inputs = _inputs(node)

    # ConditioningZeroOut blanks the conditioning: no text, whatever feeds it.
    if class_type == "ConditioningZeroOut":
        return ""

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
            if _is_ref(val):
                return get_conditioning_text(prompt, val, _seen)
        return ""

    # Combine nodes — merge text from both conditioning inputs
    if class_type in COND_COMBINE:
        parts = []
        for key in ("conditioning_1", "conditioning_2", "conditioning_to", "conditioning_from"):
            val = inputs.get(key)
            if _is_ref(val):
                t = get_conditioning_text(prompt, val, _seen)
                if t:
                    parts.append(t)
        return ", ".join(parts)

    # ControlNetApply nodes pass conditioning through slot-matched inputs:
    # output slot 0 = positive conditioning (input key "positive"),
    # output slot 1 = negative conditioning (input key "negative").
    # The deprecated single-input ControlNetApply has just ``conditioning``.
    if class_type in ("ControlNetApplyAdvanced", "ControlNetApplySD3", "ControlNetApply"):
        out_slot = cond_ref[1] if len(cond_ref) > 1 else 0
        follow_key = "negative" if out_slot == 1 else "positive"
        val = inputs.get(follow_key)
        if val is None:
            val = inputs.get("conditioning")
        if _is_ref(val):
            return get_conditioning_text(prompt, val, _seen)
        return ""

    # Unknown node type with a conditioning input — follow it anyway
    for key in ("conditioning", "conditioning_1"):
        val = inputs.get(key)
        if _is_ref(val):
            return get_conditioning_text(prompt, val, _seen)

    return ""


def resolve_model_name(prompt: dict, model_ref, _seen: set | None = None) -> str:
    """
    Walk the model edge backwards through LoRA loaders and similar passthrough nodes
    until the checkpoint loader is reached; return its ckpt_name.
    """
    if not _is_ref(model_ref):
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
    inputs = _inputs(node)

    # The name may be a widget or wired from another node (get_scalar follows it).
    # Some loaders expose the model under a different widget without being a
    # dedicated checkpoint class — honour those names when present.
    for key in ("ckpt_name", "unet_name", "model_name"):
        val = get_scalar(prompt, inputs, key)
        if isinstance(val, str) and val:
            return val
    if class_type in CHECKPOINT_CLASSES:
        return ""

    # Otherwise treat any node carrying a model edge as a passthrough (LoRA
    # loaders, model patchers, CLIP*Pip, IPAdapter, samplers-config, …) and keep
    # walking. This is deliberately generic: rather than whitelisting every
    # model-patching node, follow the ``model`` input back to the loader.
    sel = _switch_selected(prompt, class_type, inputs)
    if sel is not None:
        return resolve_model_name(prompt, sel, _seen)

    # Model merges (ModelMergeSimple, ModelMergeBlocks, …) have ``model1`` and
    # ``model2``: the name is that of the first, the second is only a fallback.
    for key in ("*", "model", "model1", "model2"):
        sub = inputs.get(key)
        if _is_ref(sub):
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
    if not _is_ref(stack_ref):
        return []
    if _seen is None:
        _seen = set()
    node_id = str(stack_ref[0])
    if node_id in _seen:
        return []
    node = _node(prompt, node_id)
    if not node or node.get("class_type") != LM_STACKER:
        return []
    inputs = _inputs(node)
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
        if _is_ref(syntax):
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

    # Core LoraLoader / LoraLoaderModelOnly (strengths may be wired from other nodes).
    if class_type in ("LoraLoader", "LoraLoaderModelOnly"):
        name = get_scalar(prompt, inputs, "lora_name")
        if isinstance(name, str) and name.lower() != "none":
            weight = get_scalar(prompt, inputs, "strength_model")
            if weight is None:
                weight = get_scalar(prompt, inputs, "strength")
            if weight is None:
                weight = 1.0
            # Strength 0 on model and clip = loader switched off (the idle
            # slot in img2prompt's gen graph): not applied, so not listed.
            clip = get_scalar(prompt, inputs, "strength_clip")
            if clip is None:
                clip = 0
            if not (weight == 0 and clip == 0):
                out.append((name, weight))

    # rgthree Lora Loader Stack: lora_1..N / strength_1..N, "None" = empty slot.
    if class_type == "Lora Loader Stack (rgthree)":
        i = 1
        while f"lora_{i}" in inputs:
            name = get_scalar(prompt, inputs, f"lora_{i}")
            if isinstance(name, str) and name.lower() != "none":
                weight = get_scalar(prompt, inputs, f"strength_{i}")
                out.append((name, 1.0 if weight is None else weight))
            i += 1

    return out


def collect_loras(prompt: dict, model_ref, _seen: set | None = None):
    """
    Walk the model edge backwards (like resolve_model_name) and collect every
    LoRA applied along the chain as (name, weight) pairs, de-duplicated by the
    filename stem (first weight wins).
    """
    if not _is_ref(model_ref):
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
    inputs = _inputs(node)

    loras = list(_node_loras(prompt, class_type, inputs))

    # Follow the selected switch branch, else ``model`` (or ``*`` on a Reroute);
    # model merges (``model1`` / ``model2``) contribute the LoRAs of both inputs.
    sel = _switch_selected(prompt, class_type, inputs)
    if _is_ref(sel):
        subs = [sel]
    elif class_type == "Reroute":
        subs = [inputs.get("*")]
    elif _is_ref(inputs.get("model")):
        subs = [inputs.get("model")]
    else:
        subs = [inputs.get("model1"), inputs.get("model2")]
    for sub in subs:
        if _is_ref(sub):
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
    if not _is_ref(val):
        return val if not isinstance(val, list) else None

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
    sub = _inputs(node)

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
    if not _is_ref(latent_ref):
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
    inputs = _inputs(node)

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
        if _is_ref(val):
            return resolve_latent_size(prompt, val, _seen)

    return None, None


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def _safe(func, default):
    """Run one field's extraction; a field that fails gets its "unresolved" value."""
    try:
        return func()
    except Exception:  # noqa: BLE001 - one broken node must not cost the other fields
        return default


def extract_from_prompt(prompt: dict) -> dict | None:
    """
    Find the primary sampler node in the ComfyUI graph and extract all relevant
    generation parameters. Returns None when no sampler node is found or the
    input is not a prompt dict.

    ``prompt`` is the parsed ComfyUI API prompt (node id -> {"class_type", "inputs"}).
    The returned dict always has these keys; a value that cannot be resolved (or
    whose extraction fails) is "" (text fields), None (scalars) or [] (loras):

    - ``positive``, ``negative``: prompt text (str)
    - ``seed``, ``steps``, ``cfg``: as stored in the graph (int / float, rarely str)
    - ``sampler_name``, ``scheduler``: str or None
    - ``model_name``: checkpoint / unet file name as in the graph (str)
    - ``width``, ``height``: latent size, None if not from an empty-latent node
    - ``loras``: list of (name, weight) tuples on the sampler's model chain
    """
    if not isinstance(prompt, dict):
        return None

    sampler_nodes = [
        (nid, node)
        for nid, node in prompt.items()
        if isinstance(node, dict)
        and isinstance(node.get("class_type"), str)
        and node["class_type"] in SAMPLER_CLASSES
    ]

    if not sampler_nodes:
        return None

    # Prefer KSampler / KSamplerAdvanced over specialised variants
    primary = next(
        ((nid, n) for nid, n in sampler_nodes if n.get("class_type") in ("KSampler", "KSamplerAdvanced")),
        sampler_nodes[0],
    )
    _, node = primary
    inputs = _inputs(node)

    # KSamplerAdvanced and its variants name the seed ``noise_seed``.
    seed_key = "noise_seed" if "noise_seed" in inputs else "seed"
    width, height = _safe(lambda: resolve_latent_size(prompt, inputs.get("latent_image")), (None, None))

    return {
        "positive": _safe(lambda: get_conditioning_text(prompt, inputs.get("positive")), ""),
        "negative": _safe(lambda: get_conditioning_text(prompt, inputs.get("negative")), ""),
        "seed": _safe(lambda: get_scalar(prompt, inputs, seed_key), None),
        "steps": _safe(lambda: get_scalar(prompt, inputs, "steps"), None),
        "cfg": _safe(lambda: get_scalar(prompt, inputs, "cfg"), None),
        "sampler_name": _safe(lambda: get_scalar(prompt, inputs, "sampler_name"), None),
        "scheduler": _safe(lambda: get_scalar(prompt, inputs, "scheduler"), None),
        "model_name": _safe(lambda: resolve_model_name(prompt, inputs.get("model")), ""),
        "width": width,
        "height": height,
        "loras": _safe(lambda: collect_loras(prompt, inputs.get("model")), []),
    }


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


def a1111_prompt_text(info: dict) -> str:
    """Positive prompt (with LoRA tags) and ``Negative prompt:`` line, no parameters line."""
    lines = []
    positive = (info.get("positive") or "").strip()
    lora_tags = _lora_tags(info)
    if lora_tags:
        positive = f"{positive} {lora_tags}".strip()
    if positive:
        lines.append(positive)

    neg = (info.get("negative") or "").strip()
    if neg:
        lines.append(f"Negative prompt: {neg}")
    return "\n".join(lines)


def a1111_params(info: dict):
    """The A1111 parameters line as ordered ``(key, value)`` string pairs (the
    keys ``Steps``, ``Sampler``, ``CFG scale``, ``Seed``, ``Size``, ``Model``)."""
    parts = []
    if info.get("steps") is not None:
        parts.append(("Steps", f"{info['steps']}"))
    if info.get("sampler_name"):
        samp = info["sampler_name"]
        if info.get("scheduler"):
            samp = f"{samp} {info['scheduler']}"
        parts.append(("Sampler", f"{samp}"))
    if info.get("cfg") is not None:
        parts.append(("CFG scale", f"{info['cfg']}"))
    if info.get("seed") is not None:
        parts.append(("Seed", f"{info['seed']}"))
    if info.get("width") and info.get("height"):
        parts.append(("Size", f"{info['width']}x{info['height']}"))
    model = model_basename(info.get("model_name"))
    if model:
        parts.append(("Model", model))
    return parts


def format_a1111(info: dict) -> str:
    """Format extracted generation fields as an A1111-style parameters string."""
    lines = []
    text = a1111_prompt_text(info)
    if text:
        lines.append(text)
    params = a1111_params(info)
    if params:
        lines.append(", ".join(f"{k}: {v}" for k, v in params))
    return "\n".join(lines)
