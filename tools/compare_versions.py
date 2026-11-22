#!/usr/bin/env python3
"""
Regression check for ``graphwalk.py``: compare two versions on a folder of images.

Loads two ``graphwalk.py`` files by path, walks a folder (recursively) of PNG /
WEBP / JPEG images, reads the ComfyUI ``prompt`` chunk of each, runs
``extract_from_prompt`` and ``format_a1111`` with both versions and compares the
results.

Only counts, file paths and the *names* of differing fields are printed, never
a value, so the output is safe to share for private image libraries.

Usage (Pillow required):

    git show main:graphwalk.py > /tmp/graphwalk_old.py
    python3 -I tools/compare_versions.py /tmp/graphwalk_old.py graphwalk.py /path/to/images

Fields compared: the keys of ``extract_from_prompt`` (``positive``, ``negative``,
``seed``, ``steps``, ``cfg``, ``sampler_name``, ``scheduler``, ``model_name``,
``width``, ``height``, ``loras``, ``clip_skip``; a version without a key counts as ``None``
there), ``None`` vs. dict (``<result>``), and the
``format_a1111`` string (``<a1111>``). A version raising an exception counts as
``<raises>``.

Exit status 0 when no image differs, 1 otherwise.

SPDX-License-Identifier: MIT
"""

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}
INFO_KEYS = (
    "positive", "negative", "seed", "steps", "cfg", "sampler_name",
    "scheduler", "model_name", "width", "height", "loras", "clip_skip",
)


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_prompt(path: Path):
    """The parsed ``prompt`` chunk of an image, or None."""
    with Image.open(path) as img:
        raw = img.info.get("prompt") if img.format == "PNG" else None
        if raw is None and img.format in ("WEBP", "JPEG"):
            exif = img.info.get("exif") or b""
            for part in exif.split(b"\x00"):
                text = part.decode("utf-8", "ignore")
                if text.lower().startswith("prompt:"):
                    raw = text.split(":", 1)[1]
                    break
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def run(module, prompt):
    """(info, a1111) or the string 'raises'."""
    try:
        info = module.extract_from_prompt(prompt)
        return info, (module.format_a1111(info) if info else None)
    except Exception:  # noqa: BLE001
        return "raises", None


def differing_fields(old, new):
    (old_info, old_text), (new_info, new_text) = old, new
    if old_info == "raises" or new_info == "raises":
        return ["<raises>"] if old_info != new_info else []
    if (old_info is None) != (new_info is None):
        return ["<result>"]
    if old_info is None:
        return []
    fields = [k for k in INFO_KEYS if old_info.get(k) != new_info.get(k)]
    if old_text != new_text:
        fields.append("<a1111>")
    return fields


def main(argv):
    if len(argv) != 4:
        print(__doc__)
        return 2
    old_mod, new_mod = load(argv[1], "gw_old"), load(argv[2], "gw_new")
    root = Path(argv[3])
    counts = {"images": 0, "no_prompt": 0, "no_sampler": 0, "same": 0, "different": 0, "error": 0}
    by_field = {}
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file()):
        counts["images"] += 1
        try:
            prompt = read_prompt(path)
        except Exception:  # noqa: BLE001 - unreadable file
            counts["error"] += 1
            continue
        if prompt is None:
            counts["no_prompt"] += 1
            continue
        old, new = run(old_mod, prompt), run(new_mod, prompt)
        if old[0] is None and new[0] is None:
            counts["no_sampler"] += 1
            continue
        fields = differing_fields(old, new)
        if not fields:
            counts["same"] += 1
            continue
        counts["different"] += 1
        for f in fields:
            by_field[f] = by_field.get(f, 0) + 1
        print(f"DIFF {path}: {', '.join(fields)}")
    print("counts:", json.dumps(counts))
    print("differing fields:", json.dumps(by_field))
    return 1 if counts["different"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
