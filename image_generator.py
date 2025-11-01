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

print(f"🔄 Using device: {DEVICE}")

# Function to load pipeline dynamically
def get_pipe(pipe_type="text2img"):
    if pipe_type == "text2img":
        pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_PATH,
            torch_dtype=DTYPE
        )
    else:
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            MODEL_PATH,
            torch_dtype=DTYPE
        )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True
    )

    # Enable memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()

    pipe = pipe.to(DEVICE)
    return pipe


def generate_image(prompt: str, init_image: Image.Image | None = None) -> str:
    """Generate an image from text or refine using img2img with highres fix."""
    save_dir = "static"
    os.makedirs(save_dir, exist_ok=True)
    img_name = f"{random.randint(1000, 9999)}.png"
    save_path = os.path.join(save_dir, img_name)

    if init_image:
        print("🎨 Running Image-to-Image with Highres Fix...")
        pipe = get_pipe("img2img")

        # Resize with moderate upscale
        init_image = init_image.convert("RGB")
        upscale = 1.3
        new_size = (int(init_image.width * upscale), int(init_image.height * upscale))
        init_image = init_image.resize(new_size, Image.Resampling.LANCZOS)

        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.4,
            num_inference_steps=30,
            guidance_scale=3.5
        )
    else:
        print(f"🖋️ Running Text-to-Image for prompt: {prompt}")
        pipe = get_pipe("text2img")

        result = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=3.5
        )

    # Save output
    image = result.images[0]
    image.save(save_path)
    print(f"✅ Generated: {save_path}")

    # Free VRAM
    del pipe
    torch.cuda.empty_cache()

    return save_path


if __name__ == "__main__":
    # Example usage
    print("✨ Example text-to-image generation...")
    path = generate_image("a futuristic cyberpunk cityscape at night, neon reflections on wet streets")
    print(f"📁 Saved to: {path}")