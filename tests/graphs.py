"""Synthetic ComfyUI API graphs for the tests (neutral prompts only)."""

import json

POSITIVE = "a red apple on a wooden table"
NEGATIVE = "blurry"


def ref(node_id, slot=0):
    return [str(node_id), slot]


def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


def basic_graph(model_ref=None, **sampler_overrides):
    """Checkpoint -> two CLIPTextEncode -> EmptyLatentImage -> KSampler."""
    sampler = dict(
        model=model_ref or ref(1),
        positive=ref(2),
        negative=ref(3),
        latent_image=ref(4),
        seed=123,
        steps=20,
        cfg=7.5,
        sampler_name="euler",
        scheduler="normal",
        denoise=1.0,
    )
    sampler.update(sampler_overrides)
    return {
        "1": node("CheckpointLoaderSimple", ckpt_name="SD15/apple_model.safetensors"),
        "2": node("CLIPTextEncode", text=POSITIVE, clip=ref(1, 1)),
        "3": node("CLIPTextEncode", text=NEGATIVE, clip=ref(1, 1)),
        "4": node("EmptyLatentImage", width=512, height=768, batch_size=1),
        "9": node("KSampler", **sampler),
    }


def with_output(graph, sampler="9", save_class="SaveImage", decode="20", save="21"):
    """Add VAEDecode and a save node behind the sampler ``sampler`` (the checkpoint's VAE is node 1, slot 2)."""
    graph[decode] = node("VAEDecode", samples=ref(sampler), vae=ref(1, 2))
    graph[save] = node(save_class, images=ref(decode), filename_prefix="apple")
    return graph


def ksampler(latent, seed=123, **overrides):
    """A KSampler on the graph of ``basic_graph`` (nodes 1, 2, 3) taking its latent from ``latent``."""
    inputs = dict(model=ref(1), positive=ref(2), negative=ref(3), latent_image=latent, seed=seed, steps=20, cfg=7.5,
                  sampler_name="euler", scheduler="normal", denoise=1.0)
    inputs.update(overrides)
    return node("KSampler", **inputs)


# Synthetic A1111 ``parameters`` chunks as Image Saver writes them (invented names and hashes).
SETTINGS_HEAD = "Steps: 20, Sampler: Euler a, CFG scale: 7.5, Seed: 123, Size: 512x768"

CIVITAI_NEW = [
    {"modelName": "Apple, Model", "modelVersionName": "v1", "modelVersionId": 111, "type": "checkpoint",
     "weight": 1.0, "air": "urn:air:sdxl:checkpoint:civitai:100@111"},
    {"modelName": "Pastel \"Soft\" Style", "modelVersionName": "v2", "modelVersionId": 222, "type": "lora",
     "weight": 0.8, "air": "urn:air:sdxl:lora:civitai:200@222"},
]
HASHES_NEW = {"model": "0a1b2c3d4e", "LORA:pastel_v1": "1122334455", "embed:soft_neg": "aabbccddee"}
LORA_HASHES_NEW = "pastel_v1: 1122334455, other_one: 66778899aa"


def _compact(value):
    return json.dumps(value, separators=(",", ":"))


# Current Image Saver: Model hash, Hashes (compact JSON), Lora hashes (quoted), Civitai resources.
PARAMS_NEW = (
    f"{POSITIVE} <lora:pastel_v1:0.8>\nNegative prompt: {NEGATIVE}\n"
    f"{SETTINGS_HEAD}, Model hash: 0a1b2c3d4e, Model: apple_model, Hashes: {_compact(HASHES_NEW)}, "
    f'Lora hashes: "{LORA_HASHES_NEW}", Version: ComfyUI, Civitai resources: {_compact(CIVITAI_NEW)}'
)

CIVITAI_OLD = [
    {"modelName": "Apple Model", "versionName": "v1", "air": "urn:air:sdxl:checkpoint:civitai:123@456"},
    {"modelName": "Soft Neg", "versionName": "v2", "air": "urn:air:sdxl:embedding:civitai:5@6", "weight": 1.0},
]

# Older Image Saver: only Civitai resources, a JSON array that is not quoted.
PARAMS_OLD = (
    f"{POSITIVE}\nNegative prompt: {NEGATIVE}\n"
    f"{SETTINGS_HEAD}, Model: apple_model, Version: ComfyUI, Civitai resources: {_compact(CIVITAI_OLD)}"
)
