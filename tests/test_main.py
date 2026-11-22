"""
Tests for the IIB adapter main.py. They need IIB (``scripts.iib``) and are skipped
without it. Run inside an IIB checkout / container, e.g.

    docker run --rm -v <this repo>:/app/plugins/comfyui_graphwalk:ro -w /app iib-local:latest \
        python -m unittest discover -s /app/plugins/comfyui_graphwalk/tests
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from scripts.iib.parsers.model import ImageGenerationInfo
    from scripts.iib.tool import parse_generation_parameters
except Exception as exc:  # noqa: BLE001 - IIB not importable here
    raise unittest.SkipTest(f"IIB not importable: {exc}")

from graphs import NEGATIVE, POSITIVE, basic_graph, node, ref  # noqa: E402

_spec = importlib.util.spec_from_file_location("graphwalk_adapter_under_test", HERE.parent / "main.py")
adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapter)


def lora_graph(name="styles/pastel_v1.5.safetensors", weight=0.8):
    g = basic_graph(model_ref=ref(6))
    g["6"] = node("LoraLoader", model=ref(1), clip=ref(1, 1), lora_name=name,
                  strength_model=weight, strength_clip=weight)
    return g


class AdapterCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.plugin = adapter.Main()

    def png(self, graph, name="a.png"):
        info = PngInfo()
        info.add_text("prompt", json.dumps(graph))
        info.add_text("workflow", "{}")
        path = self.dir / name
        Image.new("RGB", (8, 12), "white").save(path, pnginfo=info)
        return path

    def parse(self, path):
        with Image.open(path) as img:
            self.assertTrue(self.plugin.test(img, str(path)))
            return self.plugin.parse(img, str(path))

    def lora_names(self, result):
        return [x["name"] for x in result.params.extra["lora"]]


class NormalImages(AdapterCase):
    def test_meta_pos_prompt_and_lora_match_the_text_round_trip(self):
        """For a plain graph the direct meta / LoRA list equal what IIB parses from the text."""
        graph = lora_graph()
        result = self.parse(self.png(graph))
        info = adapter.extract_from_prompt(graph)
        reference = parse_generation_parameters(adapter.format_a1111(info) + ", Source Identifier: ComfyUI")
        self.assertEqual(result.params.meta, reference["meta"] | {"final_width": 8, "final_height": 12})
        # The text round-trip loses the last phrase before the LoRA tags (see Loras).
        self.assertEqual(result.params.pos_prompt, reference["pos_prompt"] + [POSITIVE])
        self.assertEqual(result.params.extra["lora"], reference["lora"])
        self.assertEqual(result.params.extra["lyco"], reference["lyco"])
        self.assertEqual(self.lora_names(result), ["pastel v1.5"])
        self.assertEqual(result.params.meta["Size-1"], "512")
        self.assertEqual(result.params.meta["Source Identifier"], "ComfyUI")

    def test_scheduler_is_its_own_meta_key(self):
        result = self.parse(self.png(basic_graph()))
        self.assertEqual(result.params.meta["Sampler"], "euler")
        self.assertEqual(result.params.meta["Schedule type"], "normal")
        self.assertEqual(adapter.Main.extra_convert_to_tag_meta_keys, [])

    def test_raw_info_is_unchanged(self):
        result = self.parse(self.png(basic_graph()))
        self.assertEqual(
            result.raw_info,
            f"{POSITIVE}\nNegative prompt: {NEGATIVE}\n"
            "Steps: 20, Sampler: euler, Schedule type: normal, CFG scale: 7.5, Seed: 123, Size: 512x768, Model: apple_model"
            ", Source Identifier: ComfyUI",
        )

    def test_few_resolved_params_keep_meta_and_clean_prompts(self):  # finding 4
        g = basic_graph(steps=None, cfg=None, seed=None, sampler_name=None, scheduler=None)
        g["4"] = node("SomeUnknownLatent")
        result = self.parse(self.png(g))
        self.assertEqual(result.params.meta["Source Identifier"], "ComfyUI")
        self.assertEqual(result.params.meta["Model"], "apple_model")
        self.assertNotIn("source identifier", result.params.pos_prompt)
        self.assertEqual(result.params.pos_prompt, ["a red apple on a wooden table"])

    def test_no_params_at_all(self):  # finding 4
        g = basic_graph(steps=None, cfg=None, seed=None, sampler_name=None, scheduler=None)
        g["4"] = node("SomeUnknownLatent")
        g["1"]["inputs"]["ckpt_name"] = ""
        result = self.parse(self.png(g))
        self.assertEqual(result.params.meta["Source Identifier"], "ComfyUI")
        self.assertEqual(result.params.pos_prompt, ["a red apple on a wooden table"])

    def test_prompt_with_colons_stays_prompt(self):
        g = basic_graph()
        g["2"]["inputs"]["text"] = "hair: red, eyes: blue, skin: pale"
        result = self.parse(self.png(g))
        self.assertEqual(result.params.pos_prompt, ["hair", "eyes", "skin"])


def old_round_trip(info):
    """What the adapter yielded before ``parse_prompt`` was used directly: IIB's text
    round-trip with a stand-in parameters line. The reference for ``PromptTags``."""
    text = adapter._core.a1111_prompt_text(info, lora_separator=", ")
    params = parse_generation_parameters(f"{text}\nSteps: 0, Sampler: none, Seed: 0" if text else "Steps: 0, Sampler: none, Seed: 0")
    return {
        "pos_prompt": params["pos_prompt"],
        "lora": adapter._merge_loras(info, params["lora"]),
        "lyco": params["lyco"],
    }


PROMPTS = [
    POSITIVE,
    "a red apple, on a wooden table",
    "hair: red, eyes: blue, skin: pale",
    "(a red apple:1.2), [on a wooden table], {soft} light, (a red apple:1.2)",
    "a red apple\nsecond line, third phrase\n  indented line , last",
    "a red apple BREAK on a wooden table BREAK soft light",
    "a red apple，on a wooden table_with-dashes，， ,",
    "A Red Apple, a red apple, A RED APPLE",
    "a red apple <lyco:glaze_v2:0.6>, on a table <lora:typed_one:0.5>, <lora:Typed_One:0.9>",
    "<lora:only_a_tag:1> a red apple",
    "embedding:easy_neg, a red apple, ext: more text",
    "   ",
    "",
]


def tag_info(positive, negative="blurry", loras=()):
    return {"positive": positive, "negative": negative, "loras": list(loras)}


class PromptTags(unittest.TestCase):
    LORA_SETS = [
        [],
        [("styles/pastel_v1.5.safetensors", 0.8)],
        [("slider+plus_v2.safetensors", -0.5), ("it's-fine.safetensors", 0.75), ("Pastel_V1.5", 1)],
    ]

    def cases(self):
        for positive in PROMPTS:
            for loras in self.LORA_SETS:
                for negative in ("", "blurry, low quality\nsecond negative line"):
                    yield tag_info(positive, negative, loras)

    def test_parse_prompt_gives_the_old_round_trip_result(self):
        for info in self.cases():
            new = adapter._prompt_tags(info)
            new["lora"] = adapter._merge_loras(info, new["lora"])
            self.assertEqual(new, old_round_trip(info), info)

    def test_old_iib_without_parse_prompt_gives_the_same_result(self):
        with mock.patch.object(adapter, "parse_prompt", None):
            for info in self.cases():
                new = adapter._prompt_tags(info)
                new["lora"] = adapter._merge_loras(info, new["lora"])
                self.assertEqual(new, old_round_trip(info), info)

    def test_fixed_expectations(self):
        info = tag_info("(a red apple:1.2), on a wooden table_top, A Red Apple", "blurry",
                        [("styles/pastel_v1.5.safetensors", 0.8)])
        self.assertEqual(adapter._prompt_tags(info), {
            "pos_prompt": ["a red apple", "on a wooden table top"],
            "lora": [{"name": "pastel v1.5", "value": 0.8}],
            "lyco": [],
        })

    def test_negative_prompt_never_becomes_a_tag(self):
        tags = adapter._prompt_tags(tag_info(POSITIVE, "blurry, ugly"))
        self.assertEqual(tags["pos_prompt"], [POSITIVE])

    def test_whole_image_matches_the_old_round_trip(self):
        for loras in self.LORA_SETS:
            g = basic_graph()
            g["2"]["inputs"]["text"] = "a red apple, (on a wooden table:1.1), soft light"
            if loras:
                g["6"] = node("LoraLoader", model=ref(1), clip=ref(1, 1), lora_name=loras[0][0],
                              strength_model=loras[0][1], strength_clip=loras[0][1])
                g["9"]["inputs"]["model"] = ref(6)
            info = adapter.extract_from_prompt(g)
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            png = PngInfo()
            png.add_text("prompt", json.dumps(g))
            path = Path(tmp.name) / "a.png"
            Image.new("RGB", (8, 12)).save(path, pnginfo=png)
            plugin = adapter.Main()
            with Image.open(path) as img:
                result = plugin.parse(img, str(path))
            old = old_round_trip(info)
            self.assertEqual(result.params.pos_prompt, old["pos_prompt"])
            self.assertEqual(result.params.extra, {"lora": old["lora"], "lyco": old["lyco"]})


class Loras(AdapterCase):
    def test_negative_weight_and_odd_names_reach_the_lora_list(self):  # finding 11
        g = basic_graph(model_ref=ref(6))
        g["6"] = node("LoraLoader", model=ref(1), clip=ref(1, 1), lora_name="slider+plus_v2.safetensors",
                      strength_model=-0.5, strength_clip=-0.5)
        g["7"] = node("LoraLoaderModelOnly", model=ref(6), lora_name="it's-fine.safetensors", strength_model=0.75)
        g["9"]["inputs"]["model"] = ref(7)
        result = self.parse(self.png(g))
        self.assertEqual(
            result.params.extra["lora"],
            [{"name": "it's fine", "value": 0.75}, {"name": "slider+plus v2", "value": -0.5}],
        )

    def test_lora_tags_stay_in_the_prompt_text(self):
        result = self.parse(self.png(lora_graph()))
        self.assertIn("<lora:pastel_v1.5:0.8>", result.raw_info)

    def test_last_phrase_before_the_lora_tags_is_a_prompt_tag(self):
        g = lora_graph()
        g["2"]["inputs"]["text"] = "a red apple, on a wooden table"
        result = self.parse(self.png(g))
        self.assertEqual(result.params.pos_prompt, ["a red apple", "on a wooden table"])
        self.assertEqual(self.lora_names(result), ["pastel v1.5"])
        self.assertIn("on a wooden table <lora:pastel_v1.5:0.8>", result.raw_info)

    def test_lora_typed_in_the_prompt_text_is_kept(self):
        g = basic_graph()
        g["2"]["inputs"]["text"] = f"{POSITIVE}, <lora:typed_one:0.5>"
        result = self.parse(self.png(g))
        self.assertEqual(self.lora_names(result), ["typed one"])

    def test_names_match_what_iib_makes_of_the_tag(self):
        for stem in ("simple", "with_underscore-and-dash", "Mixed Case v1.5", "paren (v2)", "dir_style"):
            iib = parse_generation_parameters(f"a <lora:{stem}:0.7>\nSteps: 1, Sampler: x, Seed: 1")["lora"]
            self.assertEqual(iib, [{"name": adapter._iib_lora_name(stem), "value": 0.7}], stem)


class Fallback(AdapterCase):
    def unknown_sampler_graph(self):
        g = basic_graph()
        g["9"]["class_type"] = "SamplerCustomAdvanced"
        return g

    def test_unknown_sampler_is_delegated_to_iibs_parser(self):  # finding 2
        stub = mock.Mock()
        stub.parse.return_value = ImageGenerationInfo("from iib")
        path = self.png(self.unknown_sampler_graph())
        with mock.patch.object(adapter, "_BuiltinComfyUIParser", stub):
            result = self.parse(path)
        self.assertEqual(result.raw_info, "from iib")
        stub.parse.assert_called_once()

    def test_core_exception_is_delegated_and_logged(self):  # findings 2, 15
        stub = mock.Mock()
        stub.parse.return_value = ImageGenerationInfo("from iib")
        path = self.png(basic_graph())
        with mock.patch.object(adapter, "_BuiltinComfyUIParser", stub), \
                mock.patch.object(adapter, "extract_from_prompt", side_effect=RuntimeError("boom")), \
                mock.patch.object(adapter.logger, "warning") as warn:
            result = self.parse(path)
        self.assertEqual(result.raw_info, "from iib")
        warn.assert_called_once()
        self.assertIn(str(path), warn.call_args.args)
        self.assertNotIn(POSITIVE, json.dumps([str(a) for a in warn.call_args.args]))

    def test_failing_fallback_gives_dimensions(self):  # finding 2
        stub = mock.Mock()
        stub.parse.side_effect = RuntimeError("boom")
        path = self.png(self.unknown_sampler_graph())
        with mock.patch.object(adapter, "_BuiltinComfyUIParser", stub):
            result = self.parse(path)
        self.assertEqual(result.params.meta, {"final_width": 8, "final_height": 12})

    def test_missing_fallback_gives_dimensions(self):
        path = self.png(self.unknown_sampler_graph())
        with mock.patch.object(adapter, "_BuiltinComfyUIParser", None):
            result = self.parse(path)
        self.assertEqual(result.params.meta, {"final_width": 8, "final_height": 12})

    def test_real_builtin_parser_does_not_raise(self):
        result = self.parse(self.png(self.unknown_sampler_graph()))
        self.assertIsInstance(result, ImageGenerationInfo)


class WebpExif(AdapterCase):
    def webp(self, graph, name="a.webp"):
        import piexif

        exif = piexif.dump({"0th": {
            piexif.ImageIFD.Model: ("prompt:" + json.dumps(graph)).encode(),
            piexif.ImageIFD.Make: b'workflow:{"nodes": []}',
        }})
        path = self.dir / name
        Image.new("RGB", (8, 12), "white").save(path, exif=exif)
        return path

    def test_prompt_is_read_through_piexif(self):  # finding 14
        result = self.parse(self.webp(lora_graph()))
        self.assertEqual(result.params.meta["Model"], "apple_model")
        self.assertEqual(self.lora_names(result), ["pastel v1.5"])


if __name__ == "__main__":
    unittest.main()
