#!/usr/bin/env python3
"""
Markdown lists of the node classes ``graphwalk.py`` knows, generated from its rule
tables (``SAMPLER_CLASSES``, ``SWITCHES``, ``MODEL_LOADERS``, ``OUTPUT_WIDGETS``, ...).

The README embeds the result between the markers ``<!-- supported-nodes:begin -->`` and
``<!-- supported-nodes:end -->``, so the list of supported nodes is never maintained by
hand. Standard library only.

Usage:

    python3 tools/supported_nodes.py            # print the lists
    python3 tools/supported_nodes.py --check    # exit 1 when the README differs
    python3 tools/supported_nodes.py --write    # replace the README's generated part

``--readme PATH`` and ``--core PATH`` point at other files (default: the ones next to
``tools/``). A rule table that is neither listed here nor named in ``NOT_LISTED`` is
reported by ``--check`` as well (and by ``tests/test_supported_nodes.py``), so a new
table cannot be forgotten.

SPDX-License-Identifier: MIT
"""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEGIN = "<!-- supported-nodes:begin -->"
END = "<!-- supported-nodes:end -->"

# Module-level collections of graphwalk.py that hold no node classes (field names the
# value resolver classifies by, say) and so are not listed.
NOT_LISTED = {
    "SEED_FIELDS",  # names of the inputs that hold a seed
    "STRING_FIELDS",  # names of the inputs that hold a string
}


def load_core(path=None):
    """``graphwalk.py`` loaded by file path, as a module that is not registered anywhere."""
    spec = importlib.util.spec_from_file_location("graphwalk_tables", path or ROOT / "graphwalk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Used:
    """The core module, remembering which attributes were read."""

    def __init__(self, core):
        self._core = core
        self.used = set()

    def __getattr__(self, name):
        self.used.add(name)
        return getattr(self._core, name)


def _names(items):
    return " · ".join(f"`{item}`" for item in items)


def _sorted(items):
    return sorted(items, key=lambda item: (str(item).lower(), str(item)))


def _switch_rule(rule):
    """One line saying how a known switch picks its input."""
    kind = rule.kind
    if kind in ("bool", "soft"):
        text = "`on_true` / `on_false` by `switch`"
        return text + ("; the connected one when only one is connected" if kind == "soft" else "")
    if kind == "first":
        return "the first connected `any_<n>` input, by number"
    inputs = " or ".join(f"`{prefix}<n>`" for prefix in rule.prefixes)
    if kind == "index":
        return f"{inputs} for the number `n` in `{rule.select}`"
    number = f"`{rule.select}`" if not rule.base else f"`{rule.select}` - {rule.base}"
    return f"one input `{rule.input}`, passed on only by the output slot numbered {number}"


def render(core) -> str:
    """The lists as Markdown (no trailing newline)."""
    t = core if isinstance(core, _Used) else _Used(core)
    out = []

    def section(title):
        if out:
            out.append("")
        out.append(f"#### {title}")
        out.append("")

    def bullet(text):
        out.append(f"- {text}")

    section("Samplers")
    bullet(f"KSampler-like: {_names(_sorted(t.SAMPLER_CLASSES))}")
    bullet(f"With guider, sampler, sigmas and noise as nodes: {_names(t.CUSTOM_SAMPLERS)}")
    by_kind = {kind: cls for cls, kind in t.CUSTOM_SAMPLERS.items()}
    bullet("Any other node counts as a sampler by its inputs, when it has all of")
    out.append(f"  - {_names(_sorted(t.SAMPLER_SIGNATURE))} and `seed` or `noise_seed` (as a KSampler)")
    out.append(f"  - {_names(_sorted(t.CUSTOM_SIGNATURE))} (as `{by_kind['custom']}`)")
    out.append(f"  - {_names(_sorted(t.ADVANCED_SIGNATURE))} (as `{by_kind['advanced']}`)")

    section("Output nodes")
    keys = list(dict.fromkeys(t.SAVE_CLASSES.values()))
    for key in keys:
        classes = [cls for cls, value in t.SAVE_CLASSES.items() if value == key]
        bullet(f"Save nodes taking the pictures at `{key}`: {_names(classes)}")
    bullet(f"Preview nodes: {_names(_sorted(t.PREVIEW_CLASSES))}")
    bullet(f"Inputs followed from a picture back to its latent: {_names(t.IMAGE_INPUTS + t.LATENT_INPUTS)}")

    section("Text encoders")
    for cls, keys in t.TEXT_ENCODERS.items():
        bullet(f"`{cls}`: {_names(keys)}")

    section("Conditioning")
    bullet(f"Passthrough: {_names(_sorted(t.COND_PASSTHROUGH))}")
    bullet(f"Merge, all inputs ({_names(t.COND_COMBINE_KEYS)}): {_names(_sorted(t.COND_COMBINE))}")
    bullet(f"No text, whatever feeds them: {_names(_sorted(t.COND_BLANK))}")
    first, second = t.POS_NEG_SLOTS
    bullet(
        f"Any node with both a `{first}` and a `{second}` input: output slot 0 comes from `{first}`, "
        f"slot 1 from `{second}`"
    )

    section("Custom sampler inputs")
    bullet("Guiders:")
    for cls, guider in t.GUIDERS.items():
        parts = [
            f"{label} `{key}`" if key else f"no {label}"
            for label, key in (("positive", guider.positive), ("negative", guider.negative), ("cfg", guider.cfg))
        ]
        out.append(f"  - `{cls}`: {', '.join(parts)}")
    bullet("Noise:")
    for cls, key in t.NOISE_SEEDS.items():
        out.append(f"  - `{cls}`: " + (f"seed `{key}`" if key else "no seed"))
    bullet("Samplers:")
    for cls, name in t.SAMPLER_NODES.items():
        text = "`sampler_name` widget" if name is None else f"`{name}`"
        if cls in t.SAMPLER_NOISE_DEVICE:
            text += ", with `_gpu` unless `noise_device` is `cpu`"
        out.append(f"  - `{cls}`: {text}")
    bullet("Sigmas, as schedulers (all with `steps`):")
    for cls, name in t.SCHEDULERS.items():
        out.append(f"  - `{cls}`: " + ("`scheduler` widget" if name is None else f"`{name}`"))
    bullet(f"Sigmas passed through: {_names(_sorted(t.SIGMAS_PASSTHROUGH - t.SIGMAS_RESIZERS))}")
    bullet(f"Sigmas passed through, `steps` left empty: {_names(_sorted(t.SIGMAS_RESIZERS))}")

    section("Switches")
    for cls, rule in t.SWITCHES.items():
        bullet(f"`{cls}`: {_switch_rule(rule)}")

    section("Text and values")
    for cls, key in t.TEXT_PASSTHROUGH.items():
        bullet(f"`{cls}`: text passes through from `{key}`")
    for cls, key in t.VALUE_PASSTHROUGH.items():
        bullet(f"`{cls}`: text and values pass through from `{key}`")
    bullet(
        f"Display nodes, followed through the first connected of {_names(t.SHOW_FOLLOW_KEYS)}: "
        f"{_names(_sorted(t.SHOW_PASSTHROUGH))}"
    )
    bullet(
        f"Widgets of plain value nodes: text {_names(t.TEXT_KEYS)}; seeds {_names(t.SEED_KEYS)}; "
        f"names {_names(t.STRING_KEYS)}; other numbers {_names(t.VALUE_KEYS)}"
    )
    bullet(
        f"Cached widgets of display nodes, read when the source cannot be resolved: "
        f"text {_names(t.SHOW_CACHE_KEYS[t.TEXT])}; values {_names(t.SHOW_CACHE_KEYS[t.SCALAR])}"
    )
    bullet("Multi-output nodes, output slot by slot:")
    for cls, widgets in t.OUTPUT_WIDGETS.items():
        out.append(f"  - `{cls}`: " + ", ".join(f"`{w}`" if w else "-" for w in widgets))

    section("Models")
    for widgets in dict.fromkeys(t.MODEL_LOADERS.values()):
        classes = [cls for cls, value in t.MODEL_LOADERS.items() if value == widgets]
        bullet(f"Loaders with the file name in {_names(widgets)}: {_names(classes)}")
    bullet(f"A node of any other class at the end of the model chain: {_names(t.MODEL_SOURCE_WIDGETS)}")

    section("Empty latents")
    bullet(_names(_sorted(t.LATENT_SIZE_CLASSES)))

    return "\n".join(out)


def unlisted_tables(core) -> list:
    """Module-level rule tables of ``core`` that ``render`` does not read and ``NOT_LISTED`` does not name."""
    used = _Used(core)
    render(used)
    tables = [
        name
        for name, value in vars(core).items()
        if name.isupper()
        and not name.startswith("_")
        and isinstance(value, (dict, set, frozenset, tuple, list))
    ]
    return sorted(name for name in tables if name not in used.used and name not in NOT_LISTED)


def readme_block(text: str):
    """The generated part of a README text (without the markers and the blank lines around it), or None."""
    start = text.find(BEGIN)
    end = text.find(END)
    if start < 0 or end < start:
        return None
    return text[start + len(BEGIN) : end].strip("\n")


def replace_block(text: str, block: str) -> str:
    start = text.find(BEGIN)
    end = text.find(END)
    if start < 0 or end < start:
        raise ValueError(f"the README has no {BEGIN} ... {END} pair")
    return f"{text[: start + len(BEGIN)]}\n\n{block}\n\n{text[end:]}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Markdown lists of the nodes graphwalk.py supports.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="replace the generated part of the README")
    mode.add_argument("--check", action="store_true", help="exit 1 when the README is out of date")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--core", type=Path, default=None, help="graphwalk.py to read (default: next to tools/)")
    args = parser.parse_args(argv)

    core = load_core(args.core)
    block = render(core)
    if not (args.write or args.check):
        print(block)
        return 0

    text = args.readme.read_text(encoding="utf-8")
    if args.write:
        try:
            new = replace_block(text, block)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        if new != text:
            args.readme.write_text(new, encoding="utf-8")
            print(f"updated {args.readme}")
        else:
            print(f"{args.readme} is up to date")
        return 0

    status = 0
    if readme_block(text) != block:
        print(f"{args.readme} is out of date: run python3 tools/supported_nodes.py --write", file=sys.stderr)
        status = 1
    missing = unlisted_tables(core)
    if missing:
        print(f"rule tables not in the lists (add them to render() or NOT_LISTED): {', '.join(missing)}", file=sys.stderr)
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
