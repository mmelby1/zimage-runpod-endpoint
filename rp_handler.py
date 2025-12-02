import base64
import io
import os
from pathlib import Path

import torch
from diffusers import ZImagePipeline
from PIL import Image
import runpod

# --------------------------
# Global pipeline (loaded once per worker)
# --------------------------
pipe = None


def init_pipeline():
    """Initialize and cache the Z-Image pipeline."""
    global pipe
    if pipe is not None:
        return pipe

    # Use larger RunPod volume for HF cache (fixes "no space left on device")
    cache_dir = os.getenv("HF_CACHE_DIR", "/runpod-volume/hf_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    # Allow override from env, default to Z-Image-Turbo
    model_id = os.getenv("Z_IMAGE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Z-Image pipeline per HF quickstart
    pipe_local = ZImagePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,   # change to torch.float16 if your GPU complains
        low_cpu_mem_usage=False,
    ).to(device)

    # Cache globally
    pipe = pipe_local
    return pipe


def pil_to_data_url(img: Image.Image) -> str:
    """Convert PIL image to data:image/png;base64,... string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def handler(event):
    """
    RunPod handler.

    Expected event:
    {
      "input": {
        "prompt": "...",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 9,
        "guidance_scale": 0.0,
        "seed": 42,
        "num_images": 1
      }
    }
    """
    try:
        inp = event.get("input") or {}
        prompt = inp.get("prompt")
        if not prompt:
            return {"error": "Missing 'prompt' in input."}

        height = int(inp.get("height", 1024))
        width = int(inp.get("width", 1024))
        num_inference_steps = int(inp.get("num_inference_steps", 9))
        # Z-Image-Turbo wants guidance_scale = 0.0 by default
        guidance_scale = float(inp.get("guidance_scale", 0.0))
        num_images = max(1, int(inp.get("num_images", 1)))
        seed = inp.get("seed")

        pipe = init_pipeline()

        # Figure out which device the pipe is on
        if hasattr(pipe, "_execution_device"):
            device = pipe._execution_device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if seed is not None:
            seed = int(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            # Let torch choose; we still return None in seed field
            generator = torch.Generator(device=device)

        # Batch prompts
        prompts = [prompt] * num_images

        with torch.inference_mode():
            out = pipe(
                prompt=prompts,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        images = out.images

        results = []
        for img in images:
            results.append(
                {
                    "image_url": pil_to_data_url(img),
                    "seed": seed,
                }
            )

        # RunPod expects "output" at top level
        return results

    except Exception as e:
        # Basic error info for debugging
        return {"error": str(e)}


# Start serverless worker
runpod.serverless.start({"handler": handler})
