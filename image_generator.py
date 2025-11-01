from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler
)
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

# Use Karras scheduler (similar to Exponential)
text2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    text2img_pipe.scheduler.config,
    use_karras_sigmas=True
)

# Load image-to-image pipeline
img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=DTYPE
).to(DEVICE)

img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    img2img_pipe.scheduler.config,
    use_karras_sigmas=True
)

# Optimize memory
for pipe in [text2img_pipe, img2img_pipe]:
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

print("✅ Both pipelines loaded successfully with recommended configuration!")

def generate_image(prompt: str, init_image: Image.Image | None = None) -> str:
    """Generate an image from a text prompt or refine with img2img highres fix."""
    save_dir = "static"
    os.makedirs(save_dir, exist_ok=True)
    img_name = f"{random.randint(1000, 9999)}.png"
    save_path = os.path.join(save_dir, img_name)

    if init_image:
        print("🎨 Running Image-to-Image with Highres Fix...")
        init_image = init_image.convert("RGB")
        upscale = 1.5  # upscale factor
        new_size = (int(init_image.width * upscale), int(init_image.height * upscale))
        init_image = init_image.resize(new_size, Image.Resampling.LANCZOS)

        result = img2img_pipe(
            prompt=prompt,
            image=init_image,
            strength=0.4,              # denoising for highres.fix
            num_inference_steps=30,    # recommended steps
            guidance_scale=3.5         # CFG between 2.5 and 4.5
        )
    else:
        print(f"🖋️ Running Text-to-Image for prompt: {prompt}")
        result = text2img_pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=3.5
        )

    image = result.images[0]
    image.save(save_path)
    print(f"✅ Generated: {save_path}")
    return save_path