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
