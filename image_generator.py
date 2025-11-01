from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch
import random
import os

MODEL_PATH = "./models/model.safetensors"

# Detect device
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"🔄 Loading Stable Diffusion XL pipelines on device: {DEVICE}")

# Load pipelines
text2img_pipe = StableDiffusionXLPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=DTYPE
).to(DEVICE)

img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=DTYPE
).to(DEVICE)

# Optimize for memory
for pipe in [text2img_pipe, img2img_pipe]:
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

print("✅ Both pipelines loaded successfully!")


def generate_image(prompt: str, init_image: Image.Image | None = None) -> str:
    """Generate an image from a text prompt or enhance an uploaded image."""
    save_dir = "static"
    os.makedirs(save_dir, exist_ok=True)
    img_name = f"{random.randint(1000, 9999)}.png"
    save_path = os.path.join(save_dir, img_name)

    # Common settings
    guidance_scale = 8
    num_inference_steps = 50

    if init_image:
        print("🎨 Running Image-to-Image...")
        init_image = init_image.convert("RGB")
        w, h = init_image.size

        # Resize softly if huge image
        if max(w, h) > 1024:
            factor = 1024 / max(w, h)
            init_image = init_image.resize((int(w * factor), int(h * factor)))

        # Dynamic strength (less blur for high-res)
        strength = 0.25 if max(w, h) > 800 else 0.35

        result = img2img_pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
    else:
        print(f"🖋️ Running Text-to-Image for prompt: {prompt}")
        result = text2img_pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )

    image = result.images[0]
    image.save(save_path)
    print(f"✅ Generated: {save_path}")
    return save_path