#!/usr/bin/env python3
"""
ComfyUI Graph-Walk parser plugin for sd-webui-infinite-image-browsing (IIB).

Replaces IIB's built-in ComfyUIParser, which only reads the raw KSampler inputs and
so returns wrong values on non-trivial workflows. The graph logic lives in
``graphwalk.py`` (standard library only, usable without IIB); this module is the
IIB adapter: it reads the ComfyUI ``prompt`` chunk from the image, lets the core
build an A1111-style ``parameters`` string and hands it to IIB's
``parse_generation_parameters``.

``scripts/iib/parsers/index.py`` prepends plugin instances to the parser list, so
this runs before the built-in parser and transparently supersedes it. See README.md
for behaviour, supported nodes and install steps.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Mikolas Bingemer
"""

import importlib.util
import json
import sys
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
