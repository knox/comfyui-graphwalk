#!/usr/bin/env python3
"""
ComfyUI Graph-Walk parser plugin for sd-webui-infinite-image-browsing (IIB).

Replaces IIB's built-in ComfyUIParser, which only reads the raw KSampler inputs and
so returns wrong values on non-trivial workflows. The graph logic lives in
``graphwalk.py`` (standard library only, usable without IIB); this module is the
IIB adapter: it reads the ComfyUI ``prompt`` chunk from the image, lets the core
build an A1111-style ``parameters`` string, takes the prompt tags from IIB's
``parse_generation_parameters`` and builds meta and LoRA list from the extracted
fields directly. Images the core cannot handle (no known sampler, errors) are
handed to IIB's built-in ComfyUIParser.

``scripts/iib/parsers/index.py`` prepends plugin instances to the parser list, so
this runs before the built-in parser and transparently supersedes it. See README.md
for behaviour, supported nodes and install steps.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Mikolas Bingemer
"""

import importlib.util
import json
import logging
import re
import sys
from pathlib import Path

from scripts.iib.parsers.model import ImageGenerationInfo, ImageGenerationParams
from scripts.iib.tool import (
    parse_generation_parameters,
    find,
)

try:
    from scripts.iib.logger import logger
except ImportError:
    logger = logging.getLogger(__name__)

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

# IIB's built-in ComfyUI parser: the fallback for images the core cannot handle.
try:
    from scripts.iib.parsers.comfyui import ComfyUIParser as _BuiltinComfyUIParser
except ImportError:
    _BuiltinComfyUIParser = None

# IIB depends on piexif; its parsers read WEBP / JPEG metadata through it.
try:
    import piexif
except ImportError:
    piexif = None


# ---------------------------------------------------------------------------
# Graph-walk core
# ---------------------------------------------------------------------------


def _load_core():
    """
    Load ``graphwalk.py`` from this plugin's folder. IIB loads plugins by file path
    (as module ``main``) without putting the plugin folder on ``sys.path``, so a
    plain ``import graphwalk`` would fail. A unique module name keeps it from
    colliding with other plugins or an unrelated ``graphwalk`` package.
    """
    name = "comfyui_graphwalk_core"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("graphwalk.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


_core = _load_core()
extract_from_prompt = _core.extract_from_prompt
format_a1111 = _core.format_a1111


# ---------------------------------------------------------------------------
# Reading the ComfyUI prompt chunk across image formats
# ---------------------------------------------------------------------------


def _prompt_text_from_exif(exif: bytes):
    """The ``prompt:`` JSON text in a WEBP / JPEG EXIF block, or None."""
    # Same order as IIB's own detection: structured EXIF first (piexif) …
    if piexif is not None:
        try:
            ifd0 = piexif.load(exif).get("0th", {})
        except Exception:
            ifd0 = {}
        for val in ifd0.values():
            if isinstance(val, bytes):
                text = val.decode("utf-8", "ignore")
                if text.lower().startswith("prompt:"):
                    return text.split(":", 1)[1]
    # … then a split at the null bytes …
    split = [x.decode("utf-8", "ignore") for x in exif.split(b"\x00")]
    s = find(split, lambda x: x.lower().startswith("prompt:"))
    if s:
        return s.split(":", 1)[1]
    # … then non-standard EXIF (UserComment in IFD0).
    if _extract_usercomment_from_raw_exif is not None:
        rawc = _extract_usercomment_from_raw_exif(exif)
        if rawc:
            dec = rawc.decode("utf-16", "ignore").strip("\x00") or rawc.decode("utf-8", "ignore").strip("\x00")
            p = find([dec], lambda x: x.lower().startswith("prompt:"))
            if p:
                return p.split(":", 1)[1]
    return None


def _get_prompt_dict(img):
    """Read the ComfyUI prompt chunk from PNG, WEBP, or JPEG; return the parsed dict or None."""
    raw = None
    if img.format == "PNG":
        raw = img.info.get("prompt")
    else:  # WEBP / JPEG: prompt lives in the EXIF block
        exif = img.info.get("exif")
        if exif:
            raw = _prompt_text_from_exif(exif)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# What IIB gets: meta, LoRA list, prompt tags
# ---------------------------------------------------------------------------

_RE_IMAGESIZE = re.compile(r"^(\d+)x(\d+)$")
# Characters IIB's parse_prompt removes before it looks for tags (re_parens).
_RE_PARENS = re.compile(r"[\\/\[\](){}]+")
# Three ``key: value`` pairs make IIB treat a line as the parameters line.
_PARAMS_SENTINEL = "Steps: 0, Sampler: none, Seed: 0"


def _meta_from_info(info: dict) -> dict:
    """
    IIB's ``meta`` dict, built from the extracted fields. Same keys and string
    values as ``parse_generation_parameters`` yields from the parameters line
    (an ``WxH`` value is split into ``Size-1`` / ``Size-2``), but independent of
    how many parameters could be resolved.
    """
    meta = {}
    for key, value in _core.a1111_params(info):
        size = _RE_IMAGESIZE.match(value)
        if size:
            meta[f"{key}-1"], meta[f"{key}-2"] = size.group(1), size.group(2)
        else:
            meta[key] = value
    meta["Source Identifier"] = Main.source_identifier
    return meta


def _iib_lora_name(stem: str) -> str:
    """A LoRA name the way IIB's parse_prompt turns it into a tag name."""
    return _RE_PARENS.sub("", stem.replace("-", " ").replace("_", " "))


def _merge_loras(info: dict, text_loras) -> list:
    """
    The LoRA list for IIB's indexer (``[{"name", "value"}]``): the graph's LoRAs
    first, then those IIB found as ``<lora:…>`` tags in the prompt text, de-duplicated
    by lower-case name. IIB's own tag regex misses negative weights and names with
    characters like ``+`` or ``'``, so the graph's LoRAs are not left to it.
    """
    merged, seen = [], set()
    graph = [{"name": _iib_lora_name(stem), "value": w} for stem, w in _core.lora_entries(info)]
    for entry in graph + list(text_loras or []):
        key = entry["name"].lower()
        if entry["name"].strip() and key not in seen:
            seen.add(key)
            merged.append(entry)
    return merged


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
                logger.info("ComfyUI Graph-Walk: no known sampler in %s, using IIB's ComfyUI parser", file_path)
                return self._fallback(img, file_path, width, height)
            return self._from_info(info, width, height)
        except Exception as e:
            logger.warning(
                "ComfyUI Graph-Walk: parsing %s failed (%s: %.200s), using IIB's ComfyUI parser",
                file_path, type(e).__name__, e,
            )
            return self._fallback(img, file_path, width, height)

    def _from_info(self, info, width, height) -> ImageGenerationInfo:
        a1111 = format_a1111(info)
        # "Source Identifier" must appear on the metadata line to match ComfyUIParser behaviour.
        a1111 = (a1111 + ", Source Identifier: ComfyUI") if a1111 else "Source Identifier: ComfyUI"

        # IIB only takes the last line for the parameters when it holds three
        # ``key: value`` pairs, which a graph with few resolved fields does not.
        # So IIB parses the prompt part with a stand-in parameters line (for the
        # prompt tags) and meta and LoRAs come from the extracted fields.
        # IIB splits prompt tags at commas only: without the comma the last phrase
        # and the first <lora:…> tag would form one tag, and the phrase be lost.
        prompt_text = _core.a1111_prompt_text(info, lora_separator=", ")
        params = parse_generation_parameters(f"{prompt_text}\n{_PARAMS_SENTINEL}" if prompt_text else _PARAMS_SENTINEL)
        meta = _meta_from_info(info)
        params["meta"] = meta
        params["lora"] = _merge_loras(info, params.get("lora"))
        return ImageGenerationInfo(
            a1111,
            ImageGenerationParams(
                meta=meta | {"final_width": width, "final_height": height},
                pos_prompt=params["pos_prompt"],
                extra=params,
            ),
        )

    def _fallback(self, img, file_path, width, height) -> ImageGenerationInfo:
        """IIB's built-in ComfyUI parser; on failure just the image dimensions."""
        if _BuiltinComfyUIParser is not None:
            try:
                return _BuiltinComfyUIParser.parse(img, file_path)
            except Exception as e:
                logger.warning(
                    "ComfyUI Graph-Walk: IIB's ComfyUI parser failed on %s (%s: %.200s)",
                    file_path, type(e).__name__, e,
                )
        return ImageGenerationInfo(
            params=ImageGenerationParams(meta={"final_width": width, "final_height": height})
        )
