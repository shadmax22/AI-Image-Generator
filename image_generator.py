from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch
import random
import os
import gc

MODEL_PATH = "./models/model.safetensors"

# Detect device and data type
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

# Function to safely load a pipeline with large model support
def load_pipeline(pipeline_class, model_path, device, dtype):
    try:
        pipe = pipeline_class.from_single_file(
            model_path,
            torch_dtype=dtype,
            disable_mmap=True,      # prevents 20GB mmap allocation
            device_map="auto" if device == "cuda" else None
        )
        pipe.to(device)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        print(f"✅ {pipeline_class.__name__} loaded successfully!")
        return pipe
    except Exception as e:
        print(f"❌ Failed to load {pipeline_class.__name__}: {e}")
        raise

# Load both pipelines
text2img_pipe = load_pipeline(StableDiffusionXLPipeline, MODEL_PATH, DEVICE, DTYPE)
img2img_pipe = load_pipeline(StableDiffusionXLImg2ImgPipeline, MODEL_PATH, DEVICE, DTYPE)


def generate_image(prompt: str, init_image: Image.Image | None = None) -> str:
    """Generate an image from a text prompt or enhance an uploaded image."""
    save_dir = "static"
    os.makedirs(save_dir, exist_ok=True)
    img_name = f"{random.randint(1000, 9999)}.png"
    save_path = os.path.join(save_dir, img_name)

    guidance_scale = 8
    num_inference_steps = 50

    if init_image:
        print("🎨 Running Image-to-Image...")
        init_image = init_image.convert("RGB")
        w, h = init_image.size

        # Resize safely for large inputs
        if max(w, h) > 1024:
            factor = 1024 / max(w, h)
            init_image = init_image.resize((int(w * factor), int(h * factor)))

        # Dynamic strength to reduce blur
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

    # Optional memory cleanup (useful for long runs)
    torch.cuda.empty_cache()
    gc.collect()

    return save_path


# Optional: quick test when running directly
if __name__ == "__main__":
    prompt = "a futuristic city skyline at sunset, ultra realistic, cinematic lighting"
    path = generate_image(prompt)
    print("Output saved at:", path)