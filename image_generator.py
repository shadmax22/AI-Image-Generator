from diffusers import StableDiffusionXLPipeline
import torch
import random
import os

# Load model once at import
model_path = "model.safetensors"

print("🔄 Loading Stable Diffusion XL model...")
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16
).to("mps")  # use "cuda" if NVIDIA GPU, "cpu" otherwise
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
print("✅ Model loaded successfully!")


def generate_image(prompt: str) -> str:
    """Generate image from text prompt and return file path."""
    image = pipe(
        prompt,
        num_inference_steps=15,
        guidance_scale=7.5
    ).images[0]

    img_name = f"{random.randint(1000, 9999)}.png"
    save_dir = "static"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, img_name)

    image.save(save_path)
    print(f"✅ Generated: {save_path}")
    return save_path