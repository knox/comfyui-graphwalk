"""Newer model families: text encoders, empty latents, conditioning passthrough."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import graphwalk  # noqa: E402
from graphs import POSITIVE, basic_graph, node, ref  # noqa: E402


def extract(graph):
    return graphwalk.extract_from_prompt(graph)


def with_encoder(cls, **texts):
    """basic_graph whose positive prompt is a ``cls`` node holding ``texts``."""
    g = basic_graph()
    g["2"] = node(cls, clip=ref(1, 1), **texts)
    return g


class Encoders(unittest.TestCase):
    def test_every_encoder_reads_its_text_inputs_in_order(self):
        for cls, keys in graphwalk.TEXT_ENCODERS.items():
            with self.subTest(encoder=cls):
                texts = {key: f"text {key}" for key in keys}
                self.assertEqual(extract(with_encoder(cls, **texts))["positive"], "\n".join(texts.values()))

    def test_the_encoders_of_newer_models_are_in_the_table(self):
        self.assertEqual(graphwalk.TEXT_ENCODERS["CLIPTextEncodeFlux"], ("clip_l", "t5xxl"))
        self.assertEqual(graphwalk.TEXT_ENCODERS["CLIPTextEncodeSD3"], ("clip_l", "clip_g", "t5xxl"))
        for cls in ("CLIPTextEncodeHunyuanDiT", "TextEncodeQwenImageEdit", "TextEncodeQwenImageEditPlus",
                    "TextEncodeHunyuanVideo_ImageToVideo", "CLIPTextEncodeLumina2", "CLIPTextEncodePixArtAlpha"):
            self.assertIn(cls, graphwalk.TEXT_ENCODERS)

    def test_equal_texts_count_once(self):
        g = with_encoder("CLIPTextEncodeSD3", clip_l=POSITIVE, clip_g=POSITIVE, t5xxl=POSITIVE, empty_padding="none")
        self.assertEqual(extract(g)["positive"], POSITIVE)
        g = with_encoder("CLIPTextEncodeFlux", clip_l="a red apple", t5xxl=POSITIVE, guidance=3.5)
        self.assertEqual(extract(g)["positive"], f"a red apple\n{POSITIVE}")

    def test_wired_texts(self):
        g = with_encoder("CLIPTextEncodeFlux", clip_l=ref(10), t5xxl=ref(10), guidance=3.5)
        g["10"] = node("PrimitiveString", value=POSITIVE)
        self.assertEqual(extract(g)["positive"], POSITIVE)

    def test_encoder_inputs_that_are_not_prompts_are_not_read(self):
        g = with_encoder("CLIPTextEncodeLumina2", system_prompt="superior", user_prompt=POSITIVE)
        self.assertEqual(extract(g)["positive"], POSITIVE)
        g = with_encoder("TextEncodeQwenImageEdit", prompt=POSITIVE, image=ref(5))
        g["5"] = node("LoadImage", image="apple.png")
        self.assertEqual(extract(g)["positive"], POSITIVE)

    def test_encoders_still_read_as_before(self):
        g = with_encoder("CLIPTextEncodeSDXL", text_g="global", text_l="local", width=1024, height=1024)
        self.assertEqual(extract(g)["positive"], "global\nlocal")
        g = with_encoder("smZ CLIPTextEncode", text=POSITIVE)
        self.assertEqual(extract(g)["positive"], POSITIVE)


class ConditioningPassthrough(unittest.TestCase):
    def test_flux_guidance_and_its_relatives_pass_the_text_on(self):
        for cls in ("FluxGuidance", "ReferenceLatent", "FluxKontextMultiReferenceLatentMethod",
                    "ConditioningSetAreaStrength"):
            with self.subTest(node=cls):
                self.assertIn(cls, graphwalk.COND_PASSTHROUGH)
                g = basic_graph()
                g["5"] = node(cls, conditioning=ref(2))
                g["9"]["inputs"]["positive"] = ref(5)
                self.assertEqual(extract(g)["positive"], POSITIVE)

    def test_guidance_is_not_part_of_the_result(self):
        g = basic_graph()
        g["5"] = node("FluxGuidance", conditioning=ref(2), guidance=3.5)
        g["9"]["inputs"]["positive"] = ref(5)
        self.assertEqual(
            set(extract(g)),
            {"positive", "negative", "seed", "steps", "cfg", "sampler_name", "scheduler", "model_name", "width",
             "height", "loras", "clip_skip"},
        )


class EmptyLatents(unittest.TestCase):
    def size(self, cls, **inputs):
        g = basic_graph()
        g["4"] = node(cls, **inputs)
        info = extract(g)
        return info["width"], info["height"]

    def test_every_known_class_is_a_size_source(self):
        for cls in sorted(graphwalk.LATENT_SIZE_CLASSES):
            with self.subTest(latent=cls):
                self.assertEqual(self.size(cls, width=832, height=1216, batch_size=1), (832, 1216))

    def test_empty_latent_by_name_and_inputs(self):
        self.assertEqual(self.size("EmptyFancyModelLatentVideo", width=640, height=360, length=9), (640, 360))
        self.assertEqual(self.size("EmptyLatentAudio", seconds=10, batch_size=1), (None, None))
        self.assertEqual(self.size("EmptyFancyLatent", batch_size=1), (None, None))  # no width and height
        self.assertEqual(self.size("EmptyImage", width=640, height=360), (None, None))  # not a latent
        self.assertEqual(self.size("FancyLatentThing", width=640, height=360), (None, None))  # not empty

    def test_size_through_a_new_empty_latent_and_upscale(self):
        g = basic_graph()
        g["4"] = node("EmptyFlux2LatentImage", width=1024, height=1536, batch_size=1)
        g["5"] = node("LatentUpscaleBy", samples=ref(4), upscale_method="bilinear", scale_by=2.0)
        g["9"]["inputs"]["latent_image"] = ref(5)
        info = extract(g)
        self.assertEqual((info["width"], info["height"]), (1024, 1536))


if __name__ == "__main__":
    unittest.main()
