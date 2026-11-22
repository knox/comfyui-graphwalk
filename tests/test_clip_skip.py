"""clip_skip: the layer of the CLIPSetLastLayer on the clip path of the positive prompt's text encoder."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import graphwalk  # noqa: E402
from graphs import (  # noqa: E402
    HASHES_NEW, NEGATIVE, PARAMS_NEW, POSITIVE, SETTINGS_HEAD, basic_graph, node, ref,
)
from test_custom_sampler import advanced_graph, custom_graph  # noqa: E402


def clip_skip(graph):
    return graphwalk.extract_from_prompt(graph)["clip_skip"]


def set_layer(layer, clip=None, node_id="5"):
    """A CLIPSetLastLayer node entry as ``(id, node)``, by default on the checkpoint's CLIP."""
    return node_id, node("CLIPSetLastLayer", clip=clip or ref(1, 1), stop_at_clip_layer=layer)


def with_clip(graph, *layers, positive=True):
    """Add CLIPSetLastLayer ``(id, node)`` pairs; the last one's output goes to the positive (or negative) encoder."""
    for node_id, entry in layers:
        graph[node_id] = entry
    graph["2" if positive else "3"]["inputs"]["clip"] = ref(layers[-1][0])
    return graph


class Source(unittest.TestCase):
    def test_directly_at_the_encoder(self):
        self.assertEqual(clip_skip(with_clip(basic_graph(), set_layer(-2))), 2)

    def test_layers(self):
        for layer, expected in ((-1, 1), (-2, 2), (-12, 12), (-24, 24)):
            self.assertEqual(clip_skip(with_clip(basic_graph(), set_layer(layer))), expected)

    def test_no_clip_set_last_layer_gives_none(self):
        self.assertIsNone(clip_skip(basic_graph()))

    def test_loaders_without_a_clip_input_end_the_walk(self):
        for cls in ("CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader", "CheckpointLoaderSimple"):
            g = basic_graph()
            g["7"] = node(cls, clip_name="x.safetensors")
            g["2"]["inputs"]["clip"] = ref(7)
            self.assertIsNone(clip_skip(g), cls)

    def test_only_the_positive_prompt_counts(self):
        g = with_clip(basic_graph(), set_layer(-3), positive=False)
        self.assertIsNone(clip_skip(g))
        g = with_clip(g, set_layer(-2, node_id="6"))
        self.assertEqual(clip_skip(g), 2)

    def test_the_nearest_to_the_encoder_counts(self):
        # encoder <- 6 (-4) <- 5 (-2)
        g = with_clip(basic_graph(), set_layer(-2), set_layer(-4, ref(5), "6"))
        self.assertEqual(clip_skip(g), 4)
        g = with_clip(basic_graph(), set_layer(-4), set_layer(-2, ref(5), "6"))
        self.assertEqual(clip_skip(g), 2)

    def test_first_encoder_of_a_merged_conditioning(self):
        g = basic_graph(positive=ref(8))
        g["5"] = set_layer(-2)[1]
        g["6"] = set_layer(-4, node_id="6")[1]
        g["2"]["inputs"]["clip"] = ref(5)
        g["7"] = node("CLIPTextEncode", text="a green pear", clip=ref(6))
        g["8"] = node("ConditioningCombine", conditioning_1=ref(2), conditioning_2=ref(7))
        self.assertEqual(clip_skip(g), 2)
        g["8"]["inputs"].update(conditioning_1=ref(7), conditioning_2=ref(2))
        self.assertEqual(clip_skip(g), 4)

    def test_sdxl_encoder_and_other_encoders(self):
        g = basic_graph()
        g["2"] = node("CLIPTextEncodeSDXL", text_g="a", text_l="a", clip=ref(5), width=1024, height=1024)
        g["5"] = set_layer(-2)[1]
        self.assertEqual(clip_skip(g), 2)


class Passthrough(unittest.TestCase):
    LOADERS = {
        "LoraLoader": dict(model=ref(1), lora_name="a.safetensors", strength_model=0.5, strength_clip=0.5),
        "Power Lora Loader (rgthree)": dict(model=ref(1), lora_1={"on": True, "lora": "a.safetensors", "strength": 0.5}),
        "Lora Loader Stack (rgthree)": dict(model=ref(1), lora_01="a.safetensors", strength_01=0.5),
        "Lora Loader (LoraManager)": dict(model=ref(1), text="", loras={"__value__": []}),
        "LoRA Text Loader (LoraManager)": dict(model=ref(1), lora_syntax="<lora:a:0.5>"),
    }

    def test_layer_before_the_loader(self):
        for cls, inputs in self.LOADERS.items():
            g = basic_graph()
            g["5"] = set_layer(-2)[1]
            g["6"] = node(cls, clip=ref(5), **inputs)
            g["2"]["inputs"]["clip"] = ref(6, 1)
            self.assertEqual(clip_skip(g), 2, cls)

    def test_layer_after_the_loader(self):
        for cls, inputs in self.LOADERS.items():
            g = basic_graph()
            g["6"] = node(cls, clip=ref(1, 1), **inputs)
            g["5"] = set_layer(-3, ref(6, 1))[1]
            g["2"]["inputs"]["clip"] = ref(5)
            self.assertEqual(clip_skip(g), 3, cls)

    def test_loader_without_a_layer_gives_none(self):
        for cls, inputs in self.LOADERS.items():
            g = basic_graph()
            g["6"] = node(cls, clip=ref(1, 1), **inputs)
            g["2"]["inputs"]["clip"] = ref(6, 1)
            self.assertIsNone(clip_skip(g), cls)

    def test_loader_with_the_clip_input_unconnected_gives_none(self):
        g = basic_graph()
        g["6"] = node("LoraLoader", model=ref(1), lora_name="a.safetensors", strength_model=0.5, strength_clip=0.5)
        g["2"]["inputs"]["clip"] = ref(6, 1)
        self.assertIsNone(clip_skip(g))

    def test_reroute(self):
        g = basic_graph()
        g["5"] = set_layer(-2)[1]
        g["6"] = {"class_type": "Reroute", "inputs": {"*": ref(5)}}
        g["2"]["inputs"]["clip"] = ref(6)
        self.assertEqual(clip_skip(g), 2)

    def test_switch_follows_the_selected_branch(self):
        g = basic_graph()
        g["5"] = set_layer(-2)[1]
        g["6"] = node("ComfySwitchNode", switch=True, on_true=ref(5), on_false=ref(1, 1))
        g["2"]["inputs"]["clip"] = ref(6)
        self.assertEqual(clip_skip(g), 2)
        g["6"]["inputs"]["switch"] = False
        self.assertIsNone(clip_skip(g))
        g["6"]["inputs"]["switch"] = ref(99)  # not resolved: nothing selected
        self.assertIsNone(clip_skip(g))

    def test_index_switch(self):
        g = basic_graph()
        g["5"] = set_layer(-2)[1]
        g["6"] = node("ImpactSwitch", select=2, input1=ref(1, 1), input2=ref(5))
        g["2"]["inputs"]["clip"] = ref(6)
        self.assertEqual(clip_skip(g), 2)
        g["6"]["inputs"]["select"] = 1
        self.assertIsNone(clip_skip(g))

    def test_unknown_node_with_a_clip_input_is_followed(self):
        g = basic_graph()
        g["5"] = set_layer(-2)[1]
        g["6"] = node("SomeClipPatcher", clip=ref(5), strength=0.5)
        g["2"]["inputs"]["clip"] = ref(6)
        self.assertEqual(clip_skip(g), 2)
        _, lines = graphwalk.trace_from_prompt(g)
        self.assertIn("generic: unknown node, its 'clip' input is followed", "\n".join(lines))

    def test_unknown_node_without_a_clip_input_ends_the_walk(self):
        g = basic_graph()
        g["6"] = node("SomeClipMaker", name="x")
        g["2"]["inputs"]["clip"] = ref(6)
        self.assertIsNone(clip_skip(g))

    def test_missing_node_and_cycle_give_none(self):
        g = basic_graph()
        g["2"]["inputs"]["clip"] = ref(99)
        self.assertIsNone(clip_skip(g))
        g["6"] = node("SomeClipPatcher", clip=ref(7))
        g["7"] = node("SomeClipPatcher", clip=ref(6))
        g["2"]["inputs"]["clip"] = ref(6)
        self.assertIsNone(clip_skip(g))

    def test_encoder_with_a_literal_clip_gives_none(self):
        g = basic_graph()
        g["2"]["inputs"]["clip"] = "not an edge"
        self.assertIsNone(clip_skip(g))


class Layer(unittest.TestCase):
    def test_wired_layer(self):
        g = with_clip(basic_graph(), set_layer(ref(6)))
        g["6"] = node("PrimitiveInt", value=-3)
        self.assertEqual(clip_skip(g), 3)

    def test_wired_layer_through_a_reroute(self):
        g = with_clip(basic_graph(), set_layer(ref(7)))
        g["6"] = node("PrimitiveInt", value=-2)
        g["7"] = {"class_type": "Reroute", "inputs": {"*": ref(6)}}
        self.assertEqual(clip_skip(g), 2)

    def test_unresolved_wired_layer_gives_none(self):
        self.assertIsNone(clip_skip(with_clip(basic_graph(), set_layer(ref(99)))))
        g = with_clip(basic_graph(), set_layer(ref(6)))
        g["6"] = node("SomeUnknownNode")
        self.assertIsNone(clip_skip(g))

    def test_values_that_are_no_negative_whole_number(self):
        for layer in (0, 2, 24, -1.5, "abc", None, True, False, [], float("nan"), float("inf"), float("-inf")):
            self.assertIsNone(clip_skip(with_clip(basic_graph(), set_layer(layer))), repr(layer))

    def test_whole_numbers_in_other_forms(self):
        for layer, expected in ((-2.0, 2), ("-2", 2), (" -3 ", 3)):
            self.assertEqual(clip_skip(with_clip(basic_graph(), set_layer(layer))), expected, repr(layer))

    def test_missing_widget_gives_none(self):
        g = with_clip(basic_graph(), set_layer(-2))
        del g["5"]["inputs"]["stop_at_clip_layer"]
        self.assertIsNone(clip_skip(g))


class Samplers(unittest.TestCase):
    def test_sampler_custom(self):
        g = custom_graph()
        g["5a"] = set_layer(-2)[1]
        g["2"]["inputs"]["clip"] = ref("5a")
        self.assertEqual(clip_skip(g), 2)

    def test_sampler_custom_advanced(self):
        g = advanced_graph()
        g["10"] = set_layer(-2)[1]
        g["2"]["inputs"]["clip"] = ref(10)
        self.assertEqual(clip_skip(g), 2)

    def test_advanced_with_a_basic_guider(self):
        g = advanced_graph(guider=node("BasicGuider", model=ref(1), conditioning=ref(2)))
        g["10"] = set_layer(-4)[1]
        g["2"]["inputs"]["clip"] = ref(10)
        self.assertEqual(clip_skip(g), 4)

    def test_advanced_without_a_known_guider_gives_none(self):
        g = advanced_graph(guider=node("SomeGuider", model=ref(1), positive=ref(2)))
        g["10"] = set_layer(-4)[1]
        g["2"]["inputs"]["clip"] = ref(10)
        self.assertIsNone(clip_skip(g))

    def test_positive_through_a_conditioning_passthrough(self):
        g = basic_graph(positive=ref(8))
        g["8"] = node("ConditioningSetTimestepRange", conditioning=ref(2), start=0.0, end=1.0)
        g["5"] = set_layer(-2)[1]
        g["2"]["inputs"]["clip"] = ref(5)
        self.assertEqual(clip_skip(g), 2)

    def test_the_first_pass_is_read(self):
        g = basic_graph()
        g["5"] = set_layer(-2)[1]
        g["2"]["inputs"]["clip"] = ref(5)
        g["6"] = node("CLIPTextEncode", text="refine", clip=ref(1, 1))
        g["10"] = node("KSampler", model=ref(1), positive=ref(6), negative=ref(3), latent_image=ref(9), seed=1,
                       steps=10, cfg=5, sampler_name="euler", scheduler="normal", denoise=0.4)
        g["20"] = node("VAEDecode", samples=ref(10), vae=ref(1, 2))
        g["21"] = node("SaveImage", images=ref(20), filename_prefix="apple")
        self.assertEqual(clip_skip(g), 2)  # the first pass: its positive prompt has the layer

    def test_other_keys_are_unchanged(self):
        plain = graphwalk.extract_from_prompt(basic_graph())
        skipped = graphwalk.extract_from_prompt(with_clip(basic_graph(), set_layer(-2)))
        self.assertEqual({k: v for k, v in skipped.items() if k != "clip_skip"},
                         {k: v for k, v in plain.items() if k != "clip_skip"})
        self.assertEqual((plain["clip_skip"], skipped["clip_skip"]), (None, 2))


class Trace(unittest.TestCase):
    def test_trace_shows_the_clip_path(self):
        g = basic_graph()
        g["5"] = set_layer(-2)[1]
        g["6"] = node("LoraLoader", model=ref(1), clip=ref(5), lora_name="a.safetensors", strength_model=0.5,
                      strength_clip=0.5)
        g["2"]["inputs"]["clip"] = ref(6, 1)
        info, lines = graphwalk.trace_from_prompt(g)
        self.assertEqual(info["clip_skip"], 2)
        text = "\n".join(lines)
        start = text.index("[clip_skip]")
        section = text[start:]
        for expected in (
            "· -> #2 CLIPTextEncode[0] (end)",
            "(first text encoder: #2, its 'clip' input is followed)",
            "· -> #6 LoraLoader[1]",
            "clip -> #5 CLIPSetLastLayer[0] (end)",
            "= 2",
        ):
            self.assertIn(expected, section)
        self.assertNotIn("#1 CheckpointLoaderSimple", section)  # the walk ends at the layer

    def test_trace_without_a_layer_is_none(self):
        _, lines = graphwalk.trace_from_prompt(basic_graph())
        text = "\n".join(lines)
        self.assertIn("· -> #1 CheckpointLoaderSimple[1] (end)", text.split("[clip_skip]")[1])
        self.assertTrue(text.endswith("= None"))

    def test_trace_does_not_change_the_result(self):
        g = with_clip(basic_graph(), set_layer(-2))
        self.assertEqual(graphwalk.trace_from_prompt(g)[0], graphwalk.extract_from_prompt(g))


SETTINGS = "Steps: 20, Sampler: euler, Schedule type: normal, CFG scale: 7.5, Seed: 123, Size: 512x768"


class Output(unittest.TestCase):
    def line(self, graph, resources=None):
        return graphwalk.format_a1111(graphwalk.extract_from_prompt(graph), resources).splitlines()[-1]

    def test_clip_skip_above_one_is_written_after_the_model(self):
        self.assertEqual(self.line(with_clip(basic_graph(), set_layer(-2))), f"{SETTINGS}, Model: apple_model, Clip skip: 2")
        self.assertEqual(self.line(with_clip(basic_graph(), set_layer(-12))), f"{SETTINGS}, Model: apple_model, Clip skip: 12")

    def test_clip_skip_one_and_none_are_not_written(self):
        for graph in (with_clip(basic_graph(), set_layer(-1)), basic_graph()):
            self.assertEqual(self.line(graph), f"{SETTINGS}, Model: apple_model")
        self.assertEqual(graphwalk.extract_from_prompt(with_clip(basic_graph(), set_layer(-1)))["clip_skip"], 1)

    def test_whole_text_with_clip_skip(self):
        self.assertEqual(
            graphwalk.format_a1111(graphwalk.extract_from_prompt(with_clip(basic_graph(), set_layer(-2)))),
            f"{POSITIVE}\nNegative prompt: {NEGATIVE}\n{SETTINGS}, Model: apple_model, Clip skip: 2",
        )

    def test_only_a_whole_number_above_one_is_written(self):
        info = graphwalk.extract_from_prompt(basic_graph())
        for value in (None, 0, 1, -2, True, False, "2", 2.0, [2]):
            self.assertNotIn("Clip skip", dict(graphwalk.a1111_params(info | {"clip_skip": value})), repr(value))
        self.assertIn(("Clip skip", "3"), graphwalk.a1111_params(info | {"clip_skip": 3}))
        info.pop("clip_skip")  # a dict of an older version of the core
        self.assertNotIn("Clip skip", dict(graphwalk.a1111_params(info)))

    def test_written_without_a_model_too(self):
        info = {"clip_skip": 2}
        self.assertEqual(graphwalk.a1111_params(info), [("Clip skip", "2")])

    def test_position_between_model_and_the_image_saver_resources(self):
        resources = graphwalk.image_saver_resources(PARAMS_NEW)
        params = graphwalk.a1111_params(graphwalk.extract_from_prompt(with_clip(basic_graph(), set_layer(-2))), resources)
        self.assertEqual(
            [key for key, _ in params],
            ["Steps", "Sampler", "Schedule type", "CFG scale", "Seed", "Size", "Model hash", "Model", "Clip skip",
             "Hashes", "Lora hashes", "Civitai resources"],
        )
        line = self.line(with_clip(basic_graph(), set_layer(-2)), resources)
        self.assertIn(', Model hash: 0a1b2c3d4e, Model: apple_model, Clip skip: 2, Hashes: "', line)

    def test_the_clip_skip_of_an_image_saver_chunk_is_not_taken(self):
        chunk = PARAMS_NEW.replace(SETTINGS_HEAD, f"{SETTINGS_HEAD}, Clip skip: 5")
        self.assertIn("Clip skip: 5", chunk)
        self.assertEqual(graphwalk.image_saver_resources(chunk), graphwalk.image_saver_resources(PARAMS_NEW))
        resources = graphwalk.image_saver_resources(chunk)
        self.assertNotIn("Clip skip", self.line(basic_graph(), resources))
        self.assertIn(", Clip skip: 2, ", self.line(with_clip(basic_graph(), set_layer(-2)), resources))
        self.assertNotIn("Clip skip: 5", self.line(with_clip(basic_graph(), set_layer(-2)), resources))
        self.assertEqual(resources.get("hashes"), HASHES_NEW)


if __name__ == "__main__":
    unittest.main()
