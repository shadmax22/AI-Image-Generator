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

# Load text-to-image pipeline
text2img_pipe = StableDiffusionXLPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=DTYPE
).to(DEVICE)

# Load image-to-image pipeline
img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=DTYPE
).to(DEVICE)

# Optimize memory use
text2img_pipe.enable_attention_slicing()
text2img_pipe.enable_vae_slicing()
img2img_pipe.enable_attention_slicing()
img2img_pipe.enable_vae_slicing()

print("✅ Both pipelines loaded successfully!")

def generate_image(prompt: str, init_image: Image.Image | None = None) -> str:
    """Generate an image from a text prompt or with an initial image."""
    save_dir = "static"
    os.makedirs(save_dir, exist_ok=True)
    img_name = f"{random.randint(1000, 9999)}.png"
    save_path = os.path.join(save_dir, img_name)

    if init_image:
        print("🎨 Running Image-to-Image...")
        init_image = init_image.convert("RGB").resize((512, 512))
        result = img2img_pipe(
            prompt=prompt,
            image=init_image,
            strength=0.5,
            num_inference_steps=30,
            guidance_scale=7.5
        )
    else:
        print(f"🖋️ Running Text-to-Image for prompt: {prompt}")
        result = text2img_pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        )

    image = result.images[0]
    image.save(save_path)
    print(f"✅ Generated: {save_path}")
    return save_path