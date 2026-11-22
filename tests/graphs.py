"""Synthetic ComfyUI API graphs for the tests (neutral prompts only)."""

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
