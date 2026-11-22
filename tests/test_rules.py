"""Rule tables of the walker: one test per rule class, and for each heuristic that used to guess."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import graphwalk  # noqa: E402
from graphs import NEGATIVE, POSITIVE, basic_graph, node, ref  # noqa: E402

PEAR = "a green pear on a plate"


def extract(graph):
    return graphwalk.extract_from_prompt(graph)


def with_pear(graph):
    graph["5"] = node("CLIPTextEncode", text=PEAR, clip=ref(1, 1))
    return graph


class ConditioningRules(unittest.TestCase):
    def test_generic_positive_negative_slots(self):
        # One rule for every node with both inputs, no class list.
        for cls in ("InstructPix2PixConditioning", "WanImageToVideo", "SD_4XUpscale_Conditioning"):
            g = basic_graph()
            g["5"] = node(cls, positive=ref(2), negative=ref(3), vae=ref(1, 2))
            g["9"]["inputs"]["positive"] = ref(5, 0)
            g["9"]["inputs"]["negative"] = ref(5, 1)
            info = extract(g)
            self.assertEqual((info["positive"], info["negative"]), (POSITIVE, NEGATIVE), cls)

    def test_slot_after_negative_is_not_conditioning(self):
        g = basic_graph()
        g["5"] = node("WanImageToVideo", positive=ref(2), negative=ref(3))
        g["9"]["inputs"]["positive"] = ref(5, 2)  # the latent output
        self.assertEqual(extract(g)["positive"], "")

    def test_known_passthrough_nodes(self):
        for cls in sorted(graphwalk.COND_PASSTHROUGH):
            g = basic_graph()
            g["5"] = node(cls, conditioning=ref(2))
            g["9"]["inputs"]["positive"] = ref(5)
            self.assertEqual(extract(g)["positive"], POSITIVE, cls)

    def test_unknown_node_with_conditioning_input_is_followed(self):
        g = basic_graph()
        g["5"] = node("FluxGuidance", conditioning=ref(2), guidance=3.5)
        g["9"]["inputs"]["positive"] = ref(5)
        self.assertEqual(extract(g)["positive"], POSITIVE)

    def test_unknown_merge_is_not_half_followed(self):  # was: first of conditioning_1 only
        g = basic_graph()
        g["5"] = node("SomeCustomMerge", conditioning_1=ref(2), conditioning_2=ref(3))
        g["9"]["inputs"]["positive"] = ref(5)
        self.assertEqual(extract(g)["positive"], "")

    def test_combine_nested_and_through_reroute(self):
        g = with_pear(basic_graph())
        g["6"] = node("ConditioningCombine", conditioning_1=ref(2), conditioning_2=ref(5))
        g["7"] = node("Reroute", **{"*": ref(6)})
        g["8"] = node("ConditioningConcat", conditioning_to=ref(7), conditioning_from=ref(3))
        g["9"]["inputs"]["positive"] = ref(8)
        self.assertEqual(extract(g)["positive"], f"{POSITIVE}, {PEAR}, {NEGATIVE}")

    def test_sdxl_encoder_reads_both_texts(self):
        g = basic_graph()
        g["2"] = node("CLIPTextEncodeSDXL", text_g="a red apple", text_l="on a table", clip=ref(1, 1))
        self.assertEqual(extract(g)["positive"], "a red apple\non a table")


class ModelRules(unittest.TestCase):
    def test_loader_classes_end_the_chain(self):
        for cls, widget in (
            ("UNETLoader", "unet_name"),
            ("UnetLoaderGGUF", "unet_name"),
            ("UNet loader with Name (Image Saver)", "unet_name"),
            ("Checkpoint Loader with Name (Image Saver)", "ckpt_name"),
            ("ImageOnlyCheckpointLoader", "ckpt_name"),
        ):
            g = basic_graph()
            g["1"] = node(cls, **{widget: "pear_model.safetensors"})
            self.assertEqual(extract(g)["model_name"], "pear_model.safetensors", cls)

    def test_model_name_widget_of_another_node_does_not_end_the_chain(self):
        # was: any node with a 'model_name' widget answered, here an upscaler-like node
        g = basic_graph(model_ref=ref(6))
        g["6"] = node("SomeModelPatcher", model=ref(1), model_name="upscaler.pth")
        self.assertEqual(extract(g)["model_name"], "SD15/apple_model.safetensors")
        g["6"] = node("UpscaleModelLoader", model_name="upscaler.pth")
        self.assertEqual(extract(g)["model_name"], "")

    def test_unknown_loader_is_read_by_known_widget_names(self):
        g = basic_graph()
        g["1"] = node("SomeNewLoader", ckpt_name="new_model.safetensors")
        self.assertEqual(extract(g)["model_name"], "new_model.safetensors")

    def test_unknown_node_with_model_edge_is_followed_not_read(self):
        g = basic_graph(model_ref=ref(6))
        g["6"] = node("SomePatcher", model=ref(1), ckpt_name="decoy.safetensors")
        self.assertEqual(extract(g)["model_name"], "SD15/apple_model.safetensors")

    def test_loader_with_unresolved_name_is_empty(self):
        g = basic_graph()
        g["1"]["inputs"]["ckpt_name"] = ref(7)
        g["7"] = node("SomethingElse")
        self.assertEqual(extract(g)["model_name"], "")

    def test_image_saver_name_slot_on_a_scalar_edge(self):
        g = basic_graph()
        g["1"] = node("Checkpoint Loader with Name (Image Saver)", ckpt_name="named.safetensors")
        g["7"] = node("SomeRouter", model_name=ref(1, 3))
        self.assertEqual(graphwalk.Walker(g).scalar(ref(1, 3), "model_name"), "named.safetensors")
        self.assertIsNone(graphwalk.Walker(g).scalar(ref(1, 1), "model_name"))


class SwitchRules(unittest.TestCase):
    def switch_graph(self, class_type, **inputs):
        g = with_pear(basic_graph())
        g["6"] = node(class_type, **inputs)
        g["9"]["inputs"]["positive"] = ref(6)
        return g

    def positive(self, class_type, **inputs):
        return extract(self.switch_graph(class_type, **inputs))["positive"]

    def test_unknown_switch_is_not_guessed(self):  # was: 'Switch' anywhere in the class name
        self.assertEqual(self.positive("MyCustomSwitch", select=1, input1=ref(2)), "")
        self.assertEqual(self.positive("ImpactSwitchLike", select=1, input1=ref(2)), "")

    def test_impact_switch_select_on_empty_slot_gives_nothing(self):  # was: first connected input
        self.assertEqual(self.positive("ImpactSwitch", select=2, input1=ref(2), input3=ref(5)), "")
        self.assertEqual(self.positive("ImpactSwitch", select=3, input1=ref(2), input3=ref(5)), PEAR)

    def test_impact_switch_select_missing_or_unresolved_gives_nothing(self):  # was: 1
        self.assertEqual(self.positive("ImpactSwitch", input1=ref(2)), "")
        self.assertEqual(self.positive("ImpactSwitch", select=ref(7), input1=ref(2)), "")

    def test_impact_switch_select_zero_is_not_one(self):  # was: 'value or 1'
        g = self.switch_graph("ImpactSwitch", select=ref(7), input1=ref(2))
        g["7"] = node("PrimitiveInt", value=0)
        self.assertEqual(extract(g)["positive"], "")

    def test_impact_switch_with_wired_select(self):
        g = self.switch_graph("ImpactSwitch", select=ref(7), input1=ref(2), input2=ref(5))
        g["7"] = node("PrimitiveInt", value=2)
        self.assertEqual(extract(g)["positive"], PEAR)

    def test_inversed_switch_passes_only_the_selected_output(self):
        for slot, expected in ((0, POSITIVE), (1, "")):
            g = with_pear(basic_graph())
            g["6"] = node("ImpactInversedSwitch", select=1, input=ref(2))
            g["9"]["inputs"]["positive"] = ref(6, slot)
            self.assertEqual(extract(g)["positive"], expected)

    def test_easy_index_switches_count_from_zero(self):
        self.assertEqual(self.positive("easy conditioningIndexSwitch", index=1, cond0=ref(2), cond1=ref(5)), PEAR)
        self.assertEqual(self.positive("easy conditioningIndexSwitch", index=2, cond0=ref(2), cond1=ref(5)), "")

    def test_easy_text_switch(self):
        g = basic_graph()
        g["6"] = node("easy textSwitch", input=2, text1=ref(7), text2=ref(8))
        g["7"] = node("PrimitiveString", value="a red apple")
        g["8"] = node("PrimitiveString", value="a green pear")
        g["2"]["inputs"]["text"] = ref(6)
        self.assertEqual(extract(g)["positive"], "a green pear")

    def test_easy_inversed_switch(self):
        g = basic_graph()
        g["6"] = node("easy anythingInversedSwitch", index=1, **{"in": ref(2)})
        g["9"]["inputs"]["positive"] = ref(6, 1)
        self.assertEqual(extract(g)["positive"], POSITIVE)
        g["9"]["inputs"]["positive"] = ref(6, 0)
        self.assertEqual(extract(g)["positive"], "")

    def test_latent_switch_on_the_latent_edge(self):
        g = basic_graph()
        g["6"] = node("LatentSwitch", select=1, input1=ref(4))
        g["9"]["inputs"]["latent_image"] = ref(6)
        info = extract(g)
        self.assertEqual((info["width"], info["height"]), (512, 768))
        g["6"]["inputs"]["select"] = 2
        info = extract(g)
        self.assertEqual((info["width"], info["height"]), (None, None))

    def test_switch_on_the_model_edge_selects_the_loras(self):
        g = basic_graph(model_ref=ref(8))
        g["6"] = node("LoraLoaderModelOnly", model=ref(1), lora_name="one.safetensors", strength_model=0.5)
        g["7"] = node("LoraLoaderModelOnly", model=ref(1), lora_name="two.safetensors", strength_model=0.5)
        g["8"] = node("ImpactSwitch", select=2, input1=ref(6), input2=ref(7))
        self.assertEqual(extract(g)["loras"], [("two.safetensors", 0.5)])

    def test_bool_switch_with_unresolved_flag_gives_nothing(self):  # was: the on_false branch
        g = self.switch_graph("ComfySwitchNode", switch=ref(7), on_true=ref(5), on_false=ref(2))
        g["7"] = node("SomeComparison")
        self.assertEqual(extract(g)["positive"], "")

    def test_soft_switch_with_both_branches_and_unresolved_flag_gives_nothing(self):
        g = self.switch_graph("ComfySoftSwitchNode", switch=ref(7), on_true=ref(5), on_false=ref(2))
        g["7"] = node("SomeComparison")
        self.assertEqual(extract(g)["positive"], "")

    def test_soft_switch_with_one_branch_ignores_the_flag(self):
        g = self.switch_graph("ComfySoftSwitchNode", switch=ref(7), on_true=ref(5))
        g["7"] = node("SomeComparison")
        self.assertEqual(extract(g)["positive"], PEAR)

    def test_all_switches_are_in_the_table(self):
        self.assertIn("ComfySwitchNode", graphwalk.SWITCHES)
        self.assertIn("Any Switch (rgthree)", graphwalk.SWITCHES)
        self.assertIn("ImpactSwitch", graphwalk.SWITCHES)


class ScalarRules(unittest.TestCase):
    def test_steps_do_not_come_from_a_seed_widget(self):  # was: 'seed' among the generic keys for everything
        g = basic_graph(steps=ref(5), cfg=ref(5))
        g["5"] = node("SomeSeedNode", seed=424242)
        info = extract(g)
        self.assertIsNone(info["steps"])
        self.assertIsNone(info["cfg"])

    def test_seed_comes_from_a_seed_widget(self):
        g = basic_graph(seed=ref(5))
        g["5"] = node("Seed (rgthree)", seed=424242)
        self.assertEqual(extract(g)["seed"], 424242)

    def test_noise_seed_of_advanced_sampler(self):
        g = basic_graph()
        g["9"]["class_type"] = "KSamplerAdvanced"
        g["9"]["inputs"]["noise_seed"] = ref(5)
        g["5"] = node("Seed (rgthree)", seed=77)
        self.assertEqual(extract(g)["seed"], 77)

    def test_string_fields_take_only_value(self):  # was: an int/float/seed widget could be a sampler name
        g = basic_graph(sampler_name=ref(5), scheduler=ref(6))
        g["5"] = node("PrimitiveString", value="euler")
        g["6"] = node("SomeFloat", float=0.5, **{"int": 3})
        info = extract(g)
        self.assertEqual(info["sampler_name"], "euler")
        self.assertIsNone(info["scheduler"])

    def test_value_nodes_for_numbers(self):
        g = basic_graph(steps=ref(5), cfg=ref(6))
        g["5"] = node("easy int", value=30)
        g["6"] = node("PrimitiveFloat", value=4.5)
        info = extract(g)
        self.assertEqual((info["steps"], info["cfg"]), (30, 4.5))

    def test_reroute_on_a_scalar_edge(self):
        g = basic_graph(steps=ref(6))
        g["5"] = node("PrimitiveInt", value=30)
        g["6"] = node("Reroute", **{"*": ref(5)})
        self.assertEqual(extract(g)["steps"], 30)

    def test_widget_named_like_the_field_on_an_aggregator(self):
        g = basic_graph(steps=ref(5, 1), cfg=ref(5, 2))
        g["5"] = node("SomePipe", seed=1, steps=33, cfg=2.5)
        info = extract(g)
        self.assertEqual((info["steps"], info["cfg"]), (33, 2.5))


class ImageSaverHelpers(unittest.TestCase):
    PARAMS = {"seed": 5, "steps": 28, "cfg": 3.5, "sampler": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}

    def graph(self, **sampler):
        g = basic_graph(**sampler)
        g["5"] = node("Input Parameters (Image Saver)", **self.PARAMS)
        return g

    def test_slot_to_widget_table(self):
        g = self.graph(seed=ref(5, 0), steps=ref(5, 1), cfg=ref(5, 2), sampler_name=ref(5, 3), scheduler=ref(5, 4))
        info = extract(g)
        self.assertEqual(
            (info["seed"], info["steps"], info["cfg"], info["sampler_name"], info["scheduler"]),
            (5, 28, 3.5, "dpmpp_2m", "karras"),
        )

    def test_slot_wins_over_a_widget_named_like_the_field(self):
        # KSampler.sampler_name asks for 'sampler_name'; the node's slot 3 is the widget 'sampler'.
        g = self.graph(sampler_name=ref(5, 3))
        g["5"]["inputs"]["sampler_name"] = "decoy"
        self.assertEqual(extract(g)["sampler_name"], "dpmpp_2m")

    def test_widget_wired_from_another_node(self):
        g = self.graph(seed=ref(5, 0))
        g["5"]["inputs"]["seed"] = ref(6)
        g["6"] = node("Seed (rgthree)", seed=99)
        self.assertEqual(extract(g)["seed"], 99)

    def test_selectors(self):
        g = basic_graph(sampler_name=ref(5, 1), scheduler=ref(6, 0))
        g["5"] = node("Sampler Selector (Image Saver)", sampler_name="euler_ancestral")
        g["6"] = node("Scheduler Selector (Image Saver)", scheduler="beta")
        info = extract(g)
        self.assertEqual((info["sampler_name"], info["scheduler"]), ("euler_ancestral", "beta"))

    def test_literals(self):
        g = basic_graph(steps=ref(5), cfg=ref(6))
        g["5"] = node("Int Literal (Image Saver)", **{"int": 12})
        g["6"] = node("Cfg Literal (Image Saver)", cfg=6.0)
        info = extract(g)
        self.assertEqual((info["steps"], info["cfg"]), (12, 6.0))

    def test_any_to_string_passes_its_input(self):
        g = basic_graph(sampler_name=ref(6))
        g["5"] = node("Input Parameters (Image Saver)", **self.PARAMS)
        g["6"] = node("Any to String (Image Saver)", value=ref(5, 3))
        self.assertEqual(extract(g)["sampler_name"], "dpmpp_2m")
        g = basic_graph()
        g["6"] = node("Any to String (Image Saver)", value=ref(7))
        g["7"] = node("PrimitiveString", value="a green pear")
        g["2"]["inputs"]["text"] = ref(6)
        self.assertEqual(extract(g)["positive"], "a green pear")


class DisplayNodes(unittest.TestCase):
    def show_graph(self, cache):
        g = basic_graph(sampler_name=ref(6), scheduler=ref(7))
        g["5"] = node("Input Parameters (Image Saver)", seed=1, steps=20, cfg=7.0, sampler="dpmpp_2m",
                      scheduler="karras", denoise=1.0)
        g["6"] = node("easy showAnything", anything=ref(5, 3), text=cache)
        g["7"] = node("easy showAnything", anything=ref(5, 4), text=cache)
        return g

    def test_source_wins_over_a_stale_cache(self):  # was: the cached widget came first
        info = extract(self.show_graph("euler"))
        self.assertEqual((info["sampler_name"], info["scheduler"]), ("dpmpp_2m", "karras"))

    def test_cache_is_the_fallback_when_the_source_cannot_be_resolved(self):
        g = self.show_graph("euler")
        g["5"] = node("SomethingUnknown")
        info = extract(g)
        self.assertEqual((info["sampler_name"], info["scheduler"]), ("euler", "euler"))

    def test_empty_cache_is_no_value(self):
        g = basic_graph(sampler_name=ref(6))
        g["6"] = node("easy showAnything", text="")
        self.assertIsNone(extract(g)["sampler_name"])

    def test_text_through_a_display_node_follows_the_edge(self):
        g = basic_graph()
        g["5"] = node("PrimitiveString", value="a green pear")
        g["6"] = node("ShowText|pysssss", text="a stale text", **{"value": ref(5)})
        g["2"]["inputs"]["text"] = ref(6)
        self.assertEqual(extract(g)["positive"], "a green pear")
        g["5"] = node("SomethingUnknown")
        self.assertEqual(extract(g)["positive"], "a stale text")


class SizeRules(unittest.TestCase):
    def picker_graph(self, **widgets):
        g = basic_graph()
        g["4"]["inputs"]["width"] = ref(5, 0)
        g["4"]["inputs"]["height"] = ref(5, 1)
        g["5"] = node("ResolutionPreset", **widgets)
        return g

    def size(self, g):
        info = extract(g)
        return info["width"], info["height"]

    def test_resolution_preset_label_width_first(self):
        g = self.picker_graph(preset="Portrait 2:3 · 1152×1728 · 1.9 MP", custom_width=1, custom_height=2)
        self.assertEqual(self.size(g), (1152, 1728))
        g = self.picker_graph(preset="Landscape 832x480", custom_width=1, custom_height=2)
        self.assertEqual(self.size(g), (832, 480))

    def test_resolution_preset_custom_widgets(self):
        g = self.picker_graph(preset="Custom (uses custom_width / custom_height)", custom_width=640, custom_height=960)
        self.assertEqual(self.size(g), (640, 960))

    def test_label_with_two_sizes_takes_the_first(self):
        g = self.picker_graph(preset="1024×1024 (not 512×512)")
        self.assertEqual(self.size(g), (1024, 1024))

    def test_linked_strings_are_not_scanned(self):
        g = self.picker_graph(preset=ref(6), custom_width=640, custom_height=960)
        g["6"] = node("PrimitiveString", value="Portrait 1152×1728")
        self.assertEqual(self.size(g), (640, 960))

    def test_label_heuristic_is_limited_to_size_sources(self):  # was: any node, any field
        g = basic_graph(steps=ref(5), cfg=ref(5))
        g["5"] = node("SomeNoteNode", note="rendered at 1152×1728 px")
        info = extract(g)
        self.assertIsNone(info["steps"])
        self.assertIsNone(info["cfg"])
        # the same node as a size source is read
        g["4"]["inputs"]["width"] = ref(5)
        g["4"]["inputs"]["height"] = ref(5)
        self.assertEqual(self.size(g), (1152, 1728))

    def test_plain_value_node_as_size_source(self):
        g = basic_graph()
        g["4"]["inputs"]["width"] = ref(5)
        g["5"] = node("PrimitiveInt", value=896)
        self.assertEqual(self.size(g), (896, 768))

    def test_size_through_latent_passthrough_nodes(self):
        g = basic_graph()
        g["6"] = node("Reroute", **{"*": ref(4)})
        g["7"] = node("KSampler", latent_image=ref(6))
        g["9"]["inputs"]["latent_image"] = ref(7)
        self.assertEqual(self.size(g), (512, 768))

    def test_latent_from_an_image_has_no_size(self):
        g = basic_graph()
        g["6"] = node("VAEEncode", pixels=ref(7), vae=ref(1, 2))
        g["9"]["inputs"]["latent_image"] = ref(6)
        self.assertEqual(self.size(g), (None, None))


class LoraStacks(unittest.TestCase):
    def test_lora_manager_stack_comes_first(self):
        g = basic_graph(model_ref=ref(8))
        g["6"] = node("Lora Stacker (LoraManager)", loras={"__value__": [{"name": "one", "strength": 0.5, "active": True}]})
        g["7"] = node("Lora Stacker (LoraManager)", lora_stack=ref(6),
                      loras={"__value__": [{"name": "two", "strength": 0.7, "active": True},
                                           {"name": "off", "strength": 1.0, "active": False}]})
        g["8"] = node("Lora Loader (LoraManager)", model=ref(1), clip=ref(1, 1), lora_stack=ref(7),
                      loras={"__value__": [{"name": "three", "strength": 0.9, "active": True}]})
        self.assertEqual(extract(g)["loras"], [("one", 0.5), ("two", 0.7), ("three", 0.9)])

    def test_power_lora_loader_and_stack_loader(self):
        g = basic_graph(model_ref=ref(7))
        g["6"] = node("Power Lora Loader (rgthree)", model=ref(1), clip=ref(1, 1),
                      lora_1={"on": True, "lora": "one.safetensors", "strength": 0.4},
                      lora_2={"on": False, "lora": "off.safetensors", "strength": 1.0})
        g["7"] = node("Lora Loader Stack (rgthree)", model=ref(6), lora_1="two.safetensors", strength_1=0.6,
                      lora_2="None", strength_2=1.0)
        self.assertEqual(extract(g)["loras"], [("two.safetensors", 0.6), ("one.safetensors", 0.4)])

    def test_text_loader(self):
        g = basic_graph(model_ref=ref(6))
        g["6"] = node("LoRA Text Loader (LoraManager)", model=ref(1), clip=ref(1, 1),
                      lora_syntax="<lora:one:0.5> <lora:two:0.25:0.1>")
        self.assertEqual(extract(g)["loras"], [("one", 0.5), ("two", 0.25)])


class TextRules(unittest.TestCase):
    def test_text_passthrough_and_wildcard(self):
        g = basic_graph()
        g["5"] = node("LLMPromptEnhancer", prompt=ref(6))
        g["6"] = node("ImpactWildcardProcessor", wildcard_text="__x__", populated_text="a red apple")
        g["2"]["inputs"]["text"] = ref(5)
        self.assertEqual(extract(g)["positive"], "a red apple")

    def test_local_prompt_enhancer_is_not_supported(self):
        g = basic_graph()
        g["5"] = node("LocalPromptEnhancer", prompt="a red apple", output_text="something else")
        g["2"]["inputs"]["text"] = ref(5)
        self.assertEqual(extract(g)["positive"], "")

    def test_cycle_through_a_switch_terminates(self):
        g = basic_graph()
        g["5"] = node("ImpactSwitch", select=1, input1=ref(5))
        g["2"]["inputs"]["text"] = ref(5)
        self.assertEqual(extract(g)["positive"], "")

    def test_cycle_through_merges_terminates(self):
        g = basic_graph(model_ref=ref(6))
        g["6"] = node("ModelMergeSimple", model1=ref(7), model2=ref(6))
        g["7"] = node("ModelMergeSimple", model1=ref(6), model2=ref(1))
        self.assertEqual(extract(g)["model_name"], "")


if __name__ == "__main__":
    unittest.main()
