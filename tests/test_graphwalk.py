"""Tests for the IIB-free core graphwalk.py. Run: python3 -m unittest discover tests"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import graphwalk  # noqa: E402
from graphs import NEGATIVE, POSITIVE, basic_graph, node, ref  # noqa: E402


def extract(graph):
    return graphwalk.extract_from_prompt(graph)


class Baseline(unittest.TestCase):
    def test_plain_ksampler_with_lora_exact_output(self):
        g = basic_graph(model_ref=ref(6))
        g["6"] = node(
            "LoraLoader", model=ref(1), clip=ref(1, 1),
            lora_name="styles/pastel_v1.5.safetensors", strength_model=0.8, strength_clip=0.8,
        )
        self.assertEqual(
            graphwalk.format_a1111(extract(g)),
            f"{POSITIVE} <lora:pastel_v1.5:0.8>\n"
            f"Negative prompt: {NEGATIVE}\n"
            "Steps: 20, Sampler: euler normal, CFG scale: 7.5, Seed: 123, Size: 512x768, Model: apple_model",
        )

    def test_info_keys_and_values(self):
        info = extract(basic_graph())
        self.assertEqual(
            set(info),
            {"positive", "negative", "seed", "steps", "cfg", "sampler_name", "scheduler",
             "model_name", "width", "height", "loras"},
        )
        self.assertEqual(info["positive"], POSITIVE)
        self.assertEqual(info["negative"], NEGATIVE)
        self.assertEqual((info["width"], info["height"]), (512, 768))
        self.assertEqual(info["model_name"], "SD15/apple_model.safetensors")
        self.assertEqual(info["loras"], [])

    def test_no_sampler_returns_none(self):
        g = basic_graph()
        del g["9"]
        self.assertIsNone(extract(g))
        g = {"1": node("SamplerCustomAdvanced", noise=ref(2))}
        self.assertIsNone(extract(g))


class Conditioning(unittest.TestCase):
    def test_zero_out_gives_empty_negative(self):  # finding 3
        g = basic_graph()
        g["5"] = node("ConditioningZeroOut", conditioning=ref(2))
        g["9"]["inputs"]["negative"] = ref(5)
        info = extract(g)
        self.assertEqual(info["negative"], "")
        self.assertEqual(info["positive"], POSITIVE)
        self.assertNotIn("Negative prompt", graphwalk.format_a1111(info))

    def test_deprecated_controlnet_apply_follows_conditioning(self):  # finding 5
        g = basic_graph()
        g["5"] = node("ControlNetApply", conditioning=ref(2), control_net=ref(7), image=ref(8), strength=1.0)
        g["9"]["inputs"]["positive"] = ref(5)
        self.assertEqual(extract(g)["positive"], POSITIVE)

    def test_controlnet_apply_advanced_is_slot_aware(self):
        g = basic_graph()
        g["5"] = node("ControlNetApplyAdvanced", positive=ref(2), negative=ref(3), control_net=ref(7), image=ref(8))
        g["9"]["inputs"]["positive"] = ref(5, 0)
        g["9"]["inputs"]["negative"] = ref(5, 1)
        info = extract(g)
        self.assertEqual((info["positive"], info["negative"]), (POSITIVE, NEGATIVE))

    def test_combine_joins_both(self):
        g = basic_graph()
        g["5"] = node("ConditioningCombine", conditioning_1=ref(2), conditioning_2=ref(3))
        g["9"]["inputs"]["positive"] = ref(5)
        self.assertEqual(extract(g)["positive"], f"{POSITIVE}, {NEGATIVE}")


class Sampler(unittest.TestCase):
    def test_noise_seed_on_any_sampler_variant(self):  # finding 6
        g = basic_graph()
        sampler = g["9"]
        sampler["class_type"] = "KSamplerAdvanced (Efficient)"
        sampler["inputs"]["noise_seed"] = sampler["inputs"].pop("seed")
        self.assertEqual(extract(g)["seed"], 123)

    def test_seed_on_efficient_ksampler(self):
        g = basic_graph()
        g["9"]["class_type"] = "KSampler (Efficient)"
        self.assertEqual(extract(g)["seed"], 123)

    def test_scalars_through_primitive(self):
        g = basic_graph(steps=ref(5))
        g["5"] = node("PrimitiveInt", value=31)
        self.assertEqual(extract(g)["steps"], 31)


class ModelName(unittest.TestCase):
    def test_windows_path(self):  # finding 7
        g = basic_graph()
        g["1"]["inputs"]["ckpt_name"] = "SDXL\\juggernaut.safetensors"
        self.assertTrue(graphwalk.format_a1111(extract(g)).endswith("Model: juggernaut"))

    def test_forward_slash_and_inner_dots_unchanged(self):
        self.assertEqual(graphwalk.model_basename("SD15/apple_v1.5.safetensors"), "apple_v1.5")
        self.assertEqual(graphwalk.model_basename("apple.gguf"), "apple")
        self.assertEqual(graphwalk.model_basename("mixed\\dir/apple.sft"), "apple")

    def test_linked_checkpoint_name(self):
        g = basic_graph()
        g["1"]["inputs"]["ckpt_name"] = ref(5)
        g["5"] = node("PrimitiveString", value="linked_model.safetensors")
        self.assertEqual(extract(g)["model_name"], "linked_model.safetensors")

    def test_model_merge_follows_model1(self):  # finding 12
        g = basic_graph(model_ref=ref(6))
        g["7"] = node("CheckpointLoaderSimple", ckpt_name="second_model.safetensors")
        g["6"] = node("ModelMergeSimple", model1=ref(1), model2=ref(7), ratio=0.5)
        self.assertEqual(extract(g)["model_name"], "SD15/apple_model.safetensors")

    def test_merge_collects_loras_of_both_inputs(self):  # finding 12
        g = basic_graph(model_ref=ref(6))
        g["7"] = node("CheckpointLoaderSimple", ckpt_name="second_model.safetensors")
        g["10"] = node("LoraLoaderModelOnly", model=ref(1), lora_name="one.safetensors", strength_model=0.5)
        g["11"] = node("LoraLoaderModelOnly", model=ref(7), lora_name="two.safetensors", strength_model=-0.25)
        g["12"] = node("LoraLoaderModelOnly", model=ref(10), lora_name="one.safetensors", strength_model=0.9)
        g["6"] = node("ModelMergeBlocks", model1=ref(12), model2=ref(11), input=1.0, middle=1.0, out=1.0)
        self.assertEqual(extract(g)["loras"], [("one.safetensors", 0.9), ("two.safetensors", -0.25)])


class Loras(unittest.TestCase):
    def lora_graph(self, **lora):
        g = basic_graph(model_ref=ref(6))
        g["6"] = node("LoraLoader", model=ref(1), clip=ref(1, 1), lora_name="one.safetensors", **lora)
        return g

    def test_linked_strength_is_resolved(self):  # finding 10
        g = self.lora_graph(strength_model=ref(7), strength_clip=ref(8))
        g["7"] = node("PrimitiveFloat", value=0.6)
        g["8"] = node("PrimitiveFloat", value=0.4)
        self.assertEqual(extract(g)["loras"], [("one.safetensors", 0.6)])

    def test_linked_zero_strengths_switch_the_loader_off(self):  # finding 10
        g = self.lora_graph(strength_model=ref(7), strength_clip=ref(7))
        g["7"] = node("PrimitiveFloat", value=0)
        self.assertEqual(extract(g)["loras"], [])

    def test_literal_zero_strengths_switch_the_loader_off(self):
        self.assertEqual(extract(self.lora_graph(strength_model=0, strength_clip=0))["loras"], [])

    def test_negative_weight_tag(self):
        text = graphwalk.format_a1111(extract(self.lora_graph(strength_model=-0.5, strength_clip=-0.5)))
        self.assertIn("<lora:one:-0.5>", text)

    def test_lora_entries(self):
        info = {"loras": [("dir/two words (v2).safetensors", "0.75"), ("x.pt", None)]}
        self.assertEqual(graphwalk.lora_entries(info), [("two words (v2)", 0.75), ("x", 1.0)])


class Switches(unittest.TestCase):
    def switch_graph(self, class_type, **inputs):
        g = basic_graph()
        g["5"] = node("CLIPTextEncode", text="a green pear on a plate", clip=ref(1, 1))
        g["6"] = node(class_type, **inputs)
        g["9"]["inputs"]["positive"] = ref(6)
        return g

    def test_comfy_switch_true_and_false(self):  # finding 13
        for flag, expected in ((True, "a green pear on a plate"), (False, POSITIVE)):
            g = self.switch_graph("ComfySwitchNode", switch=flag, on_true=ref(5), on_false=ref(2))
            self.assertEqual(extract(g)["positive"], expected)

    def test_comfy_switch_linked_flag(self):  # finding 13
        g = self.switch_graph("ComfySwitchNode", switch=ref(7), on_true=ref(5), on_false=ref(2))
        g["7"] = node("PrimitiveBoolean", value=True)
        self.assertEqual(extract(g)["positive"], "a green pear on a plate")

    def test_soft_switch_uses_the_connected_branch(self):  # finding 13
        g = self.switch_graph("ComfySoftSwitchNode", switch=True, on_false=ref(2))
        self.assertEqual(extract(g)["positive"], POSITIVE)

    def test_hard_switch_without_selected_branch_has_no_text(self):
        g = self.switch_graph("ComfySwitchNode", switch=True, on_false=ref(2))
        self.assertEqual(extract(g)["positive"], "")

    def test_any_switch_takes_first_connected_input(self):  # finding 13
        g = self.switch_graph("Any Switch (rgthree)", any_01=None, any_02=ref(5), any_03=ref(2))
        self.assertEqual(extract(g)["positive"], "a green pear on a plate")

    def test_any_switch_goes_by_slot_number_not_key_order(self):
        g = self.switch_graph("Any Switch (rgthree)", any_03=ref(2), any_02=ref(5))
        self.assertEqual(extract(g)["positive"], "a green pear on a plate")

    def test_any_switch_on_model_edge(self):  # finding 13
        g = basic_graph(model_ref=ref(6))
        g["6"] = node("Any Switch (rgthree)", any_01=ref(1))
        self.assertEqual(extract(g)["model_name"], "SD15/apple_model.safetensors")

    def test_impact_switch_select(self):
        g = self.switch_graph("ImpactSwitch", select=2, input1=ref(2), input2=ref(5))
        self.assertEqual(extract(g)["positive"], "a green pear on a plate")


class TextNodes(unittest.TestCase):
    def text_graph(self, class_type, **inputs):
        g = basic_graph()
        g["5"] = node(class_type, **inputs)
        g["2"]["inputs"]["text"] = ref(5)
        return g

    def positive(self, g):
        return extract(g)["positive"]

    def test_string_function_append_tidy_yes_joins_with_comma(self):  # finding 8
        g = self.text_graph("StringFunction|pysssss", action="append", tidy_tags="yes",
                            text_a="a red apple", text_b="", text_c="on a table")
        self.assertEqual(self.positive(g), "a red apple, on a table")

    def test_string_function_append_tidy_no_joins_plainly(self):  # finding 8
        g = self.text_graph("StringFunction|pysssss", action="append", tidy_tags="no",
                            text_a="a red apple ", text_b="on a table")
        self.assertEqual(self.positive(g), "a red apple on a table")

    def test_string_function_replace(self):  # finding 8
        g = self.text_graph("StringFunction|pysssss", action="replace", tidy_tags="no",
                            text_a="a red apple", text_b="red", text_c="green")
        self.assertEqual(self.positive(g), "a green apple")

    def test_string_function_replace_regex(self):  # finding 8
        g = self.text_graph("StringFunction|pysssss", action="replace", tidy_tags="no",
                            text_a="a red apple", text_b="/r.d/", text_c="green")
        self.assertEqual(self.positive(g), "a green apple")

    def test_string_function_tidy_step(self):  # finding 8
        g = self.text_graph("StringFunction|pysssss", action="replace", tidy_tags="yes",
                            text_a="a  red apple ,, on   a table ", text_b="zzz", text_c="")
        self.assertEqual(self.positive(g), "a red apple, on a table")

    def test_string_concatenate_defaults_to_empty_delimiter_and_keeps_empty_parts(self):
        g = self.text_graph("StringConcatenate", string_a="a red apple", string_b="on a table")
        self.assertEqual(self.positive(g), "a red appleon a table")
        g = self.text_graph("StringConcatenate", string_a="a red apple", string_b="", delimiter=", ")
        self.assertEqual(self.positive(g), "a red apple, ")

    def test_string_concatenate_linked_empty_delimiter_stays_empty(self):
        g = self.text_graph("StringConcatenate", string_a="a", string_b="b", delimiter=ref(6))
        g["6"] = node("PrimitiveString", value="")
        self.assertEqual(self.positive(g), "ab")

    def test_regex_replace(self):
        g = self.text_graph("RegexReplace", string=POSITIVE, regex_pattern="red", replace="green")
        self.assertEqual(self.positive(g), "a green apple on a wooden table")

    def test_regex_replace_with_linked_widgets(self):  # finding 9
        g = self.text_graph("RegexReplace", string=POSITIVE, regex_pattern=ref(6), replace="green",
                            count=ref(7), case_insensitive=ref(8))
        g["6"] = node("PrimitiveString", value="RED")
        g["7"] = node("PrimitiveInt", value=1)
        g["8"] = node("PrimitiveBoolean", value=True)
        self.assertEqual(self.positive(g), "a green apple on a wooden table")

    def test_regex_replace_bad_pattern_keeps_text(self):
        g = self.text_graph("RegexReplace", string=POSITIVE, regex_pattern="(", replace="x")
        self.assertEqual(self.positive(g), POSITIVE)


class Robustness(unittest.TestCase):
    def test_non_dict_input(self):  # finding 9
        for bad in (None, [], "text", 5, [["1", 0]]):
            self.assertIsNone(graphwalk.extract_from_prompt(bad))

    def test_garbage_nodes_are_ignored(self):  # finding 9
        g = basic_graph()
        g["20"] = "not a node"
        g["21"] = {"class_type": ["x"], "inputs": None}
        self.assertEqual(extract(g)["positive"], POSITIVE)

    def test_empty_ref_lists(self):  # finding 9
        g = basic_graph(steps=[])
        g["9"]["inputs"]["model"] = []
        g["9"]["inputs"]["positive"] = []
        info = extract(g)
        self.assertEqual((info["positive"], info["model_name"], info["steps"]), ("", "", None))

    def test_failing_field_does_not_cost_the_others(self):  # finding 9
        g = basic_graph()
        orig = graphwalk.Walker.cond_text

        def boom(self, cond_ref, seen=frozenset()):
            if cond_ref == ref(2):
                raise RuntimeError("boom")
            return orig(self, cond_ref, seen)

        graphwalk.Walker.cond_text = boom
        try:
            info = extract(g)
        finally:
            graphwalk.Walker.cond_text = orig
        self.assertEqual(info["positive"], "")
        self.assertEqual(info["negative"], NEGATIVE)
        self.assertEqual(info["steps"], 20)

    def test_cycles_terminate(self):
        g = basic_graph()
        g["5"] = node("Reroute", **{"*": ref(6)})
        g["6"] = node("Reroute", **{"*": ref(5)})
        g["9"]["inputs"]["positive"] = ref(5)
        self.assertEqual(extract(g)["positive"], "")


class FormatParts(unittest.TestCase):
    def test_params_are_ordered_string_pairs(self):
        info = extract(basic_graph())
        self.assertEqual(
            graphwalk.a1111_params(info),
            [("Steps", "20"), ("Sampler", "euler normal"), ("CFG scale", "7.5"), ("Seed", "123"),
             ("Size", "512x768"), ("Model", "apple_model")],
        )

    def test_no_params_no_line(self):
        self.assertEqual(graphwalk.format_a1111({"positive": "a red apple"}), "a red apple")
        self.assertEqual(graphwalk.format_a1111({}), "")


if __name__ == "__main__":
    unittest.main()
