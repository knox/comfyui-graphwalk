"""Which sampler a graph is read from: the search from the output nodes, the fallback, the signature."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import graphwalk  # noqa: E402
from graphs import basic_graph, ksampler, node, ref, with_output  # noqa: E402


def extract(graph):
    return graphwalk.extract_from_prompt(graph)


def trace(graph):
    return "\n".join(graphwalk.trace_from_prompt(graph)[1])


def two_pass_graph():
    """Empty latent -> KSampler 9 (seed 1) -> LatentUpscaleBy -> KSampler 11 (seed 2) -> decode -> save."""
    g = basic_graph(seed=1)
    g["10"] = node("LatentUpscaleBy", samples=ref(9), upscale_method="bilinear", scale_by=1.5)
    g["11"] = ksampler(ref(10), seed=2, denoise=0.5)
    return with_output(g, sampler="11")


class OutputSearch(unittest.TestCase):
    def test_the_sampler_behind_the_save_node_wins_over_a_stray_one(self):
        g = basic_graph(seed=1)
        g = {"5": ksampler(ref(4), seed=99), **g}  # first in the dict, but nothing saves its result
        self.assertEqual(extract(g)["seed"], 99)  # no output node: the old rule
        g["31"] = node("VAEDecode", samples=ref(9), vae=ref(1, 2))
        g["32"] = node("SaveImage", images=ref(31))
        self.assertEqual(extract(g)["seed"], 1)

    def test_latent_hires_fix_gives_the_first_pass(self):
        info = extract(two_pass_graph())
        self.assertEqual((info["seed"], info["steps"], info["sampler_name"]), (1, 20, "euler"))
        self.assertEqual((info["width"], info["height"]), (512, 768))

    def test_pixel_space_hires_fix_gives_the_first_pass(self):
        g = basic_graph(seed=1)
        g["10"] = node("VAEDecode", samples=ref(9), vae=ref(1, 2))
        g["11"] = node("ImageScale", image=ref(10), upscale_method="lanczos", width=768, height=1152, crop="disabled")
        g["12"] = node("VAEEncode", pixels=ref(11), vae=ref(1, 2))
        g["13"] = ksampler(ref(12), seed=2, denoise=0.4)
        info = extract(with_output(g, sampler="13"))
        self.assertEqual(info["seed"], 1)
        self.assertEqual((info["width"], info["height"]), (512, 768))

    def test_any_empty_latent_by_name_starts_the_first_pass(self):
        g = two_pass_graph()
        g["4"] = node("Empty Latent (Image Saver)", resolution="1:1", width_override=0, height_override=0)
        info = extract(g)
        self.assertEqual(info["seed"], 1)
        self.assertEqual((info["width"], info["height"]), (None, None))  # a size source needs width and height

    def test_refiner_pair_gives_the_base_sampler(self):
        g = basic_graph()
        g["9"] = node("KSamplerAdvanced", model=ref(1), positive=ref(2), negative=ref(3), latent_image=ref(4),
                      add_noise="enable", noise_seed=11, steps=25, cfg=7, sampler_name="euler", scheduler="normal",
                      start_at_step=0, end_at_step=20, return_with_leftover_noise="enable")
        g["10"] = node("KSamplerAdvanced", model=ref(1), positive=ref(2), negative=ref(3), latent_image=ref(9),
                       add_noise="disable", noise_seed=22, steps=25, cfg=7, sampler_name="euler", scheduler="normal",
                       start_at_step=20, end_at_step=10000, return_with_leftover_noise="disable")
        self.assertEqual(extract(with_output(g, sampler="10"))["seed"], 11)

    def test_img2img_gives_the_sampler_nearest_the_output(self):
        g = basic_graph()
        g["5"] = node("LoadImage", image="apple.png")
        g["6"] = node("VAEEncode", pixels=ref(5), vae=ref(1, 2))
        g["9"] = ksampler(ref(6), seed=1, denoise=0.6)
        g["10"] = ksampler(ref(9), seed=2, denoise=0.3)
        info = extract(with_output(g, sampler="10"))
        self.assertEqual(info["seed"], 2)
        self.assertEqual((info["width"], info["height"]), (None, None))
        self.assertIn("nearest to the output", trace(with_output(g, sampler="10")))

    def test_several_save_nodes_the_lowest_id_by_number_decides(self):
        g = two_pass_graph()  # saved at 21 from sampler 11, which leads to 9 at the start
        g["12"] = ksampler(ref(4), seed=7)
        g["13"] = node("VAEDecode", samples=ref(12), vae=ref(1, 2))
        g["14"] = node("SaveImage", images=ref(13))
        self.assertEqual(extract(g)["seed"], 7)  # save node 14 comes before 21; 14 < 21 as numbers
        g["9"]["inputs"]["seed"] = 5
        g["8"] = node("SaveImage", images=ref(20))  # the first pass again; 8 < 14, though "8" > "14" as text
        self.assertEqual(extract(g)["seed"], 5)

    def test_node_id_order_is_numeric(self):
        ids = ["12", "9", "5:3", "5:12", "100", "b"]
        self.assertEqual(sorted(ids, key=graphwalk._id_key), ["5:3", "5:12", "9", "12", "100", "b"])

    def test_a_save_node_without_a_sampler_is_skipped(self):
        g = basic_graph()
        g["5"] = node("LoadImage", image="apple.png")
        g["6"] = node("SaveImage", images=ref(5))
        g = with_output(g)
        self.assertEqual(extract(g)["seed"], 123)

    def test_save_nodes_come_before_previews(self):
        g = basic_graph(seed=1)
        g["12"] = ksampler(ref(4), seed=2)
        g["13"] = node("VAEDecode", samples=ref(12), vae=ref(1, 2))
        g["14"] = node("PreviewImage", images=ref(13))  # sorts before the save node below
        g = with_output(g, sampler="9", decode="30", save="31")
        self.assertEqual(extract(g)["seed"], 1)
        del g["31"]
        self.assertEqual(extract(g)["seed"], 2)  # only the preview is left

    def test_unknown_output_node_with_images_input_counts_as_save(self):
        g = with_output(basic_graph(), save_class="SomeVideoCombine")
        self.assertIn("output #21 SomeVideoCombine", trace(g))
        self.assertIn("no output node", trace({**g, "21": node("SomeVideoCombine")}))  # nothing wired: fallback

    def test_node_with_images_input_used_by_another_node_is_not_an_output(self):
        g = basic_graph(seed=1)
        g["12"] = ksampler(ref(4), seed=2)
        g["13"] = node("VAEDecode", samples=ref(12), vae=ref(1, 2))
        g["14"] = node("ImageSharpen", images=ref(13))  # takes images, but 15 uses it
        g["15"] = node("ImageInvert", image=ref(14))
        g = {**with_output(g, sampler="9", decode="30", save="31"), "16": node("SaveImage", images=ref(15))}
        text = trace(g)
        self.assertNotIn("output #14", text)
        self.assertEqual(extract(g)["seed"], 2)  # 16 comes before 31

    def test_detailer_between_decode_and_save_is_passed(self):
        g = basic_graph()
        g["5"] = node("FaceDetailer", image=ref(20), model=ref(1), positive=ref(2), negative=ref(3), seed=777,
                      steps=10, cfg=4, sampler_name="euler", scheduler="normal", denoise=0.4)
        g = with_output(g)
        g["21"]["inputs"]["images"] = ref(5)
        self.assertEqual(extract(g)["seed"], 123)

    def test_video_output_goes_through_create_video(self):
        g = basic_graph()
        g["20"] = node("VAEDecode", samples=ref(9), vae=ref(1, 2))
        g["21"] = node("CreateVideo", images=ref(20), fps=16.0)
        g["22"] = node("SaveVideo", video=ref(21), filename_prefix="apple", format="auto", codec="auto")
        self.assertIn("output #22 SaveVideo", trace(g))
        self.assertIn("first pass", trace(g))

    def test_every_save_class_names_its_input(self):
        for cls, key in graphwalk.SAVE_CLASSES.items():
            with self.subTest(cls=cls):
                g = basic_graph()
                g["20"] = node("VAEDecode", samples=ref(9), vae=ref(1, 2))
                g["21"] = node(cls, **{key: ref(20)})
                self.assertIn(f"output #21 {cls}", trace(g))
                self.assertIn("first pass", trace(g))

    def test_cycle_on_the_way_back_terminates(self):
        g = basic_graph()
        g["10"] = node("LatentUpscaleBy", samples=ref(9))
        g["9"]["inputs"]["latent_image"] = ref(10)
        info = extract(with_output(g))
        self.assertEqual(info["seed"], 123)
        self.assertEqual((info["width"], info["height"]), (None, None))


class Fallback(unittest.TestCase):
    def test_no_output_node_uses_the_first_ksampler(self):
        g = {"5": ksampler(ref(4), seed=99), **basic_graph()}
        self.assertEqual(extract(g)["seed"], 99)
        self.assertIn("no output node leads to a sampler", trace(g))

    def test_ksampler_comes_before_its_variants(self):
        g = basic_graph()
        g = {"5": node("BNK_Unsampler", model=ref(1), steps=5, end_at_step=3, cfg=1, sampler_name="euler",
                       scheduler="normal", normalize="disable", positive=ref(2), negative=ref(3), latent_image=ref(4)),
             **g}
        self.assertEqual(extract(g)["seed"], 123)

    def test_output_nodes_that_lead_nowhere_fall_back(self):
        g = basic_graph()
        g["5"] = node("LoadImage", image="apple.png")
        g["6"] = node("SaveImage", images=ref(5))
        self.assertEqual(extract(g)["seed"], 123)

    def test_graph_with_a_sampler_but_no_output_is_not_empty(self):
        self.assertIsNotNone(extract(basic_graph()))


class Signature(unittest.TestCase):
    def fancy(self, **overrides):
        inputs = dict(model=ref(1), positive=ref(2), negative=ref(3), latent_image=ref(4), seed=5, steps=8, cfg=3)
        inputs.update(overrides)
        return node("FancySampler", **inputs)

    def test_unknown_class_with_a_sampler_signature_is_found_from_the_output(self):
        g = basic_graph()
        g["9"] = self.fancy()
        info = extract(with_output(g))
        self.assertEqual((info["seed"], info["steps"], info["cfg"]), (5, 8, 3))
        self.assertEqual(info["positive"], "a red apple on a wooden table")

    def test_noise_seed_counts_as_seed(self):
        g = basic_graph()
        g["9"] = self.fancy(noise_seed=6)
        del g["9"]["inputs"]["seed"]
        self.assertEqual(extract(with_output(g))["seed"], 6)

    def test_signature_is_found_without_an_output_too(self):
        g = basic_graph()
        g["9"] = self.fancy()
        self.assertEqual(extract(g)["seed"], 5)

    def test_missing_inputs_do_not_make_a_sampler(self):
        for missing in ("model", "positive", "negative", "latent_image", "steps", "seed"):
            with self.subTest(missing=missing):
                g = basic_graph()
                g["9"] = self.fancy()
                del g["9"]["inputs"][missing]
                self.assertIsNone(extract(with_output(g)))

    def test_known_classes_win_the_fallback_over_a_signature(self):
        g = basic_graph()
        g = {"5": self.fancy(seed=8), **g}
        self.assertEqual(extract(g)["seed"], 123)


class TraceOfTheChoice(unittest.TestCase):
    def test_trace_shows_output_path_sampler_and_reason(self):
        text = trace(two_pass_graph())
        for expected in (
            "output #21 SaveImage",
            "· -> #20 VAEDecode[0]",
            "samples -> #11 KSampler[0]",
            "latent_image -> #10 LatentUpscaleBy[0]",
            "latent_image -> #4 EmptyLatentImage[0] (end)",
            "sampler #9 KSampler (first pass: its latent starts at #4 EmptyLatentImage)",
        ):
            self.assertIn(expected, text)

    def test_trace_does_not_change_the_choice(self):
        g = two_pass_graph()
        self.assertEqual(graphwalk.trace_from_prompt(g)[0], extract(g))


if __name__ == "__main__":
    unittest.main()
