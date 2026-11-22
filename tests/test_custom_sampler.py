"""SamplerCustom and SamplerCustomAdvanced: guider, noise, sampler and sigmas nodes."""

import sys
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import graphwalk  # noqa: E402
from graphs import NEGATIVE, POSITIVE, basic_graph, node, ref, with_output  # noqa: E402


def extract(graph):
    return graphwalk.extract_from_prompt(graph)


def base():
    """Checkpoint (1), two prompts (2, 3), empty latent (4); no sampler yet."""
    g = basic_graph()
    del g["9"]
    return g


def custom_graph(**overrides):
    """SamplerCustom (9) with KSamplerSelect (5) and BasicScheduler (6)."""
    g = base()
    g["5"] = node("KSamplerSelect", sampler_name="dpmpp_2m")
    g["6"] = node("BasicScheduler", model=ref(1), scheduler="karras", steps=30, denoise=1.0)
    inputs = dict(model=ref(1), add_noise=True, noise_seed=42, cfg=6.5, positive=ref(2), negative=ref(3),
                  sampler=ref(5), sigmas=ref(6), latent_image=ref(4))
    inputs.update(overrides)
    g["9"] = node("SamplerCustom", **inputs)
    return g


def advanced_graph(guider=None, noise=None, sampler=None, sigmas=None):
    """SamplerCustomAdvanced (9) with RandomNoise (5), CFGGuider (6), KSamplerSelect (7), BasicScheduler (8)."""
    g = base()
    g["5"] = noise or node("RandomNoise", noise_seed=42)
    g["6"] = guider or node("CFGGuider", model=ref(1), positive=ref(2), negative=ref(3), cfg=3.5)
    g["7"] = sampler or node("KSamplerSelect", sampler_name="euler")
    g["8"] = sigmas or node("BasicScheduler", model=ref(1), scheduler="simple", steps=28, denoise=1.0)
    g["9"] = node("SamplerCustomAdvanced", noise=ref(5), guider=ref(6), sampler=ref(7), sigmas=ref(8),
                  latent_image=ref(4))
    return g


class SamplerCustom(unittest.TestCase):
    def test_all_fields(self):
        info = extract(custom_graph())
        self.assertEqual(
            info,
            {"positive": POSITIVE, "negative": NEGATIVE, "seed": 42, "steps": 30, "cfg": 6.5,
             "sampler_name": "dpmpp_2m", "scheduler": "karras", "model_name": "SD15/apple_model.safetensors",
             "width": 512, "height": 768, "loras": [], "clip_skip": None},
        )

    def test_without_noise_there_is_no_seed(self):
        self.assertIsNone(extract(custom_graph(add_noise=False))["seed"])

    def test_wired_values(self):
        g = custom_graph(cfg=ref(10), noise_seed=ref(11), sampler=ref(12))
        g["10"] = node("PrimitiveFloat", value=4.5)
        g["11"] = node("PrimitiveInt", value=77)
        g["12"] = node("Reroute", **{"*": ref(5)})
        info = extract(g)
        self.assertEqual((info["cfg"], info["seed"], info["sampler_name"]), (4.5, 77, "dpmpp_2m"))

    def test_found_from_the_output_with_its_size(self):
        info = extract(with_output(custom_graph()))
        self.assertEqual((info["seed"], info["width"], info["height"]), (42, 512, 768))

    def test_unknown_sigmas_leave_steps_and_scheduler_empty(self):
        g = custom_graph()
        g["6"] = node("SomethingUnknown", steps=30)
        info = extract(g)
        self.assertEqual((info["steps"], info["scheduler"]), (None, None))
        self.assertEqual(info["positive"], POSITIVE)

    def test_found_by_signature(self):
        g = custom_graph()
        g["9"]["class_type"] = "MySampler"
        self.assertEqual(extract(g)["sampler_name"], "dpmpp_2m")


class SamplerCustomAdvanced(unittest.TestCase):
    def test_all_fields(self):
        info = extract(advanced_graph())
        self.assertEqual(
            info,
            {"positive": POSITIVE, "negative": NEGATIVE, "seed": 42, "steps": 28, "cfg": 3.5,
             "sampler_name": "euler", "scheduler": "simple", "model_name": "SD15/apple_model.safetensors",
             "width": 512, "height": 768, "loras": [], "clip_skip": None},
        )

    def test_found_from_the_output_and_by_signature(self):
        g = advanced_graph()
        self.assertEqual(extract(with_output(g))["seed"], 42)
        g["9"]["class_type"] = "MyAdvancedSampler"
        self.assertEqual(extract(g)["steps"], 28)

    def test_disable_noise_has_no_seed(self):
        self.assertIsNone(extract(advanced_graph(noise=node("DisableNoise")))["seed"])

    def test_unknown_noise_has_no_seed(self):
        self.assertIsNone(extract(advanced_graph(noise=node("SomeNoise", noise_seed=5)))["seed"])

    def test_wired_seed_steps_and_cfg(self):
        g = advanced_graph(
            noise=node("RandomNoise", noise_seed=ref(10)),
            guider=node("CFGGuider", model=ref(1), positive=ref(2), negative=ref(3), cfg=ref(11)),
            sigmas=node("BasicScheduler", model=ref(1), scheduler="simple", steps=ref(12), denoise=1.0),
        )
        g["10"] = node("PrimitiveInt", value=99)
        g["11"] = node("PrimitiveFloat", value=2.5)
        g["12"] = node("PrimitiveInt", value=18)
        info = extract(g)
        self.assertEqual((info["seed"], info["cfg"], info["steps"]), (99, 2.5, 18))

    def test_reroute_and_switch_on_the_new_edges(self):
        g = advanced_graph()
        g["10"] = node("Reroute", **{"*": ref(6)})
        g["11"] = node("ImpactSwitch", select=2, input1=ref(13), input2=ref(10))
        g["13"] = node("CFGGuider", model=ref(1), positive=ref(3), negative=ref(2), cfg=9)
        g["9"]["inputs"]["guider"] = ref(11)
        info = extract(g)
        self.assertEqual((info["positive"], info["negative"], info["cfg"]), (POSITIVE, NEGATIVE, 3.5))

    def test_guiders(self):
        cond = lambda **kw: node("CLIPTextEncode", text=kw.get("text", POSITIVE), clip=ref(1, 1))  # noqa: E731
        cases = {
            "CFGGuider": (dict(model=ref(1), positive=ref(2), negative=ref(3), cfg=3.5), NEGATIVE, 3.5),
            "BasicGuider": (dict(model=ref(1), conditioning=ref(2)), "", None),
            "DualCFGGuider": (dict(model=ref(1), cond1=ref(2), cond2=ref(10), negative=ref(3), cfg_conds=5.0,
                                   cfg_cond2_negative=2.0, style="regular"), NEGATIVE, None),
            "DualModelGuider": (dict(model=ref(1), model_negative=ref(1), positive=ref(2), negative=ref(3), cfg=4.0),
                                NEGATIVE, 4.0),
            "PerpNegGuider": (dict(model=ref(1), positive=ref(2), negative=ref(3), empty_conditioning=ref(10),
                                   cfg=4.5, neg_scale=1.0), NEGATIVE, 4.5),
            "LTXVDualCFGGuider": (dict(model=ref(1), positive=ref(2), negative=ref(3), video_cfg=3.0, audio_cfg=7.0),
                                  NEGATIVE, 3.0),
            "ScheduledCFGGuidance": (dict(model=ref(1), positive=ref(2), negative=ref(3), cfg=6.0, start_percent=0.0,
                                          end_percent=1.0), NEGATIVE, 6.0),
        }
        self.assertEqual(set(cases), set(graphwalk.GUIDERS))
        for cls, (inputs, negative, cfg) in cases.items():
            with self.subTest(guider=cls):
                g = advanced_graph(guider=node(cls, **inputs))
                g["10"] = cond(text="something else")
                info = extract(g)
                self.assertEqual(info["positive"], POSITIVE)
                self.assertEqual((info["negative"], info["cfg"]), (negative, cfg))
                self.assertEqual(info["model_name"], "SD15/apple_model.safetensors")

    def test_unknown_guider_reads_nothing_but_the_scheduler_model_gives_the_checkpoint(self):
        g = advanced_graph(guider=node("SomeGuider", model=ref(1), positive=ref(2), negative=ref(3), cfg=3.5))
        info = extract(g)
        self.assertEqual((info["positive"], info["negative"], info["cfg"]), ("", "", None))
        self.assertEqual(info["model_name"], "SD15/apple_model.safetensors")
        self.assertEqual(info["steps"], 28)

    def test_model_comes_from_the_guider_with_its_loras(self):
        g = advanced_graph(guider=node("CFGGuider", model=ref(10), positive=ref(2), negative=ref(3), cfg=3.5))
        g["10"] = node("LoraLoaderModelOnly", model=ref(1), lora_name="styles/pastel.safetensors", strength_model=0.6)
        info = extract(g)
        self.assertEqual(info["loras"], [("styles/pastel.safetensors", 0.6)])
        self.assertEqual(info["model_name"], "SD15/apple_model.safetensors")

    def test_hires_with_two_advanced_samplers_gives_the_first_pass(self):
        g = advanced_graph()
        g["10"] = node("LatentUpscaleBy", samples=ref(9), upscale_method="bilinear", scale_by=1.5)
        g["11"] = node("RandomNoise", noise_seed=7)
        g["12"] = node("SamplerCustomAdvanced", noise=ref(11), guider=ref(6), sampler=ref(7), sigmas=ref(8),
                       latent_image=ref(10))
        info = extract(with_output(g, sampler="12"))
        self.assertEqual((info["seed"], info["width"], info["height"]), (42, 512, 768))

    def test_trace_names_the_nodes_behind_the_inputs(self):
        _, lines = graphwalk.trace_from_prompt(advanced_graph())
        text = "\n".join(lines)
        for expected in ("sampler #9 SamplerCustomAdvanced", "guider: #6 CFGGuider", "noise: #5 RandomNoise",
                         "sampler: #7 KSamplerSelect", "sigmas: #8 BasicScheduler"):
            self.assertIn(expected, text)


class Samplers(unittest.TestCase):
    def test_sampler_nodes(self):
        cases = {
            "KSamplerSelect": ({"sampler_name": "heun"}, "heun"),
            "SamplerEulerAncestral": ({"eta": 1.0, "s_noise": 1.0}, "euler_ancestral"),
            "SamplerEulerAncestralCFGPP": ({"eta": 1.0, "s_noise": 1.0}, "euler_ancestral_cfg_pp"),
            "SamplerDPMPP_2S_Ancestral": ({"eta": 1.0, "s_noise": 1.0}, "dpmpp_2s_ancestral"),
            "SamplerLMS": ({"order": 4}, "lms"),
            "SamplerDPMAdaptative": ({"order": 3}, "dpm_adaptive"),
            "SamplerER_SDE": ({"solver_type": "ER-SDE", "max_stage": 3, "eta": 1.0, "s_noise": 1.0}, "er_sde"),
            "SamplerDPMPP_3M_SDE": ({"eta": 1.0, "s_noise": 1.0, "noise_device": "gpu"}, "dpmpp_3m_sde_gpu"),
            "SamplerDPMPP_2M_SDE": ({"solver_type": "midpoint", "noise_device": "cpu"}, "dpmpp_2m_sde"),
            "SamplerDPMPP_SDE": ({"eta": 1.0, "noise_device": "gpu"}, "dpmpp_sde_gpu"),
        }
        self.assertEqual(set(cases), set(graphwalk.SAMPLER_NODES))
        for cls, (inputs, expected) in cases.items():
            with self.subTest(sampler=cls):
                self.assertEqual(extract(advanced_graph(sampler=node(cls, **inputs)))["sampler_name"], expected)

    def test_unresolved_noise_device_leaves_the_name_empty(self):
        g = advanced_graph(sampler=node("SamplerDPMPP_2M_SDE", solver_type="midpoint", noise_device=ref(10)))
        g["10"] = node("SomethingUnknown")
        self.assertIsNone(extract(g)["sampler_name"])

    def test_unknown_sampler_node_is_empty(self):
        self.assertIsNone(extract(advanced_graph(sampler=node("SamplerSASolver", eta=1.0)))["sampler_name"])


class Schedulers(unittest.TestCase):
    def test_every_scheduler_class(self):
        extra = {
            "BasicScheduler": dict(model=ref(1), scheduler="beta", denoise=1.0),
            "BetaSamplingScheduler": dict(model=ref(1), alpha=0.6, beta=0.6),
            "SDTurboScheduler": dict(model=ref(1), denoise=1.0),
            "AlignYourStepsScheduler": dict(model_type="SDXL", denoise=1.0),
            "GITSScheduler": dict(coeff=1.2, denoise=1.0),
            "OptimalStepsScheduler": dict(model_type="FLUX", denoise=1.0),
            "Flux2Scheduler": dict(width=1024, height=1024),
            "Ideogram4Scheduler": dict(width=1024, height=1024, mu=0.0, std=1.75),
        }
        names = {cls: name for cls, name in graphwalk.SCHEDULERS.items() if name is not None}
        names["BasicScheduler"] = "beta"
        self.assertEqual(set(names), set(graphwalk.SCHEDULERS))
        for cls, name in names.items():
            with self.subTest(scheduler=cls):
                info = extract(advanced_graph(sigmas=node(cls, steps=12, **extra.get(cls, {}))))
                self.assertEqual((info["steps"], info["scheduler"]), (12, name))

    def test_the_names_of_comfyuis_own_schedulers(self):
        # the same words KSampler's scheduler widget uses
        for cls, name in (("KarrasScheduler", "karras"), ("ExponentialScheduler", "exponential"),
                          ("BetaSamplingScheduler", "beta")):
            self.assertEqual(graphwalk.SCHEDULERS[cls], name)

    def test_same_steps_pass_through_flip_and_first_sigma(self):
        for cls in ("FlipSigmas", "SetFirstSigma"):
            with self.subTest(passthrough=cls):
                g = advanced_graph(sigmas=node("KarrasScheduler", steps=15, sigma_max=14.6, sigma_min=0.03, rho=7.0))
                g["10"] = node(cls, sigmas=ref(8), sigma=100.0)
                g["9"]["inputs"]["sigmas"] = ref(10)
                info = extract(g)
                self.assertEqual((info["steps"], info["scheduler"]), (15, "karras"))

    def test_resizing_passthrough_leaves_the_steps_empty(self):
        for cls, inputs in (("SplitSigmas", {"step": 10}), ("SplitSigmasDenoise", {"denoise": 0.5}),
                            ("ExtendIntermediateSigmas", {"steps": 2})):
            with self.subTest(passthrough=cls):
                g = advanced_graph(sigmas=node("KarrasScheduler", steps=20, sigma_max=14.6, sigma_min=0.03, rho=7.0))
                g["10"] = node(cls, sigmas=ref(8), **inputs)
                g["9"]["inputs"]["sigmas"] = ref(10, 1)
                info = extract(g)
                self.assertEqual((info["steps"], info["scheduler"]), (None, "karras"))

    def test_sigmas_through_a_reroute(self):
        g = advanced_graph()
        g["10"] = node("Reroute", **{"*": ref(8)})
        g["9"]["inputs"]["sigmas"] = ref(10)
        self.assertEqual(extract(g)["steps"], 28)

    def test_manual_sigmas_are_not_a_scheduler(self):
        info = extract(advanced_graph(sigmas=node("ManualSigmas", sigmas="1, 0.5, 0")))
        self.assertEqual((info["steps"], info["scheduler"]), (None, None))


class NewerModelGraph(unittest.TestCase):
    def test_flux_style_graph(self):
        g = {
            "1": node("UNETLoader", unet_name="flux/apple_flux.safetensors", weight_dtype="default"),
            "2": node("ModelSamplingFlux", model=ref(1), max_shift=1.15, base_shift=0.5, width=1024, height=1024),
            "3": node("LoraLoaderModelOnly", model=ref(2), lora_name="styles/pastel.safetensors", strength_model=0.7),
            "4": node("CLIPTextEncodeFlux", clip=ref(9), clip_l=POSITIVE, t5xxl=POSITIVE, guidance=3.5),
            "5": node("FluxGuidance", conditioning=ref(4), guidance=3.5),
            "6": node("BasicGuider", model=ref(3), conditioning=ref(5)),
            "7": node("EmptyFlux2LatentImage", width=1024, height=1536, batch_size=1),
            "8": node("Flux2Scheduler", steps=20, width=1024, height=1536),
            "10": node("RandomNoise", noise_seed=1234),
            "11": node("KSamplerSelect", sampler_name="euler"),
            "12": node("SamplerCustomAdvanced", noise=ref(10), guider=ref(6), sampler=ref(11), sigmas=ref(8),
                       latent_image=ref(7)),
            "13": node("VAEDecode", samples=ref(12), vae=ref(9)),
            "14": node("SaveImage", images=ref(13)),
        }
        info = extract(g)
        self.assertEqual(
            info,
            {"positive": POSITIVE, "negative": "", "seed": 1234, "steps": 20, "cfg": None,
             "sampler_name": "euler", "scheduler": "flux2", "model_name": "flux/apple_flux.safetensors",
             "width": 1024, "height": 1536, "loras": [("styles/pastel.safetensors", 0.7)],
             "clip_skip": None},
        )
        self.assertEqual(
            graphwalk.format_a1111(info),
            f"{POSITIVE} <lora:pastel:0.7>\n"
            "Steps: 20, Sampler: euler, Schedule type: flux2, Seed: 1234, Size: 1024x1536, Model: apple_flux",
        )

    def test_model_patchers_pass_the_model_on(self):
        for cls in ("ModelSamplingFlux", "ModelSamplingSD3", "ModelSamplingAuraFlow", "CFGNorm", "FreeU_V2"):
            with self.subTest(patcher=cls):
                g = advanced_graph(
                    guider=node("CFGGuider", model=ref(10), positive=ref(2), negative=ref(3), cfg=3.5),
                    sigmas=node("BasicScheduler", model=ref(10), scheduler="simple", steps=28, denoise=1.0),
                )
                g["10"] = node(cls, model=ref(11))
                g["11"] = node("LoraLoaderModelOnly", model=ref(1), lora_name="a.safetensors", strength_model=1.0)
                info = extract(g)
                self.assertEqual(info["model_name"], "SD15/apple_model.safetensors")
                self.assertEqual(info["loras"], [("a.safetensors", 1.0)])


if __name__ == "__main__":
    unittest.main()


class BrokenSamplerInputs(unittest.TestCase):
    def test_unreadable_sampler_inputs_give_none(self):
        g = {"9": node("SamplerCustomAdvanced", noise=ref(1), guider=ref(2), sampler=ref(3), sigmas=ref(4),
                       latent_image=ref(5))}
        with mock.patch.object(graphwalk, "_advanced_sources", side_effect=RuntimeError("broken")):
            self.assertIsNone(graphwalk.extract_from_prompt(g))
