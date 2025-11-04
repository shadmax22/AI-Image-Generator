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

# Detect device and dtype
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

# ------------ MODEL INITIALIZATION (ON LOAD ONLY) ------------
def init_pipe(pipe_type="text2img"):
    print(f"⚙️ Initializing {pipe_type} pipeline...")
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
        algorithm_type="dpmsolver++",
        solver_order=3,
        use_karras_sigmas=False  # Simple (beta upscale) scheduler
    )

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()

    pipe = pipe.to(DEVICE)
    print(f"✅ {pipe_type} pipeline ready.")
    return pipe


# Load once globally
TEXT2IMG_PIPE = init_pipe("text2img")
IMG2IMG_PIPE = init_pipe("img2img")

def print_vram():
    """Print current VRAM usage."""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"📊 VRAM used: {used:.2f} GB / {total:.2f} GB")


def generate_image(prompt: str, init_image: Image.Image | None = None) -> str:
    """Generate image from prompt or refine uploaded image."""
    save_dir = "static"
    os.makedirs(save_dir, exist_ok=True)
    img_name = f"{random.randint(1000, 9999)}.png"
    save_path = os.path.join(save_dir, img_name)

    print_vram()

    if init_image:
        print("🎨 Running Image-to-Image with Highres Fix...")
        pipe = IMG2IMG_PIPE

        init_image = init_image.convert("RGB")

        # Resize large images
        MAX_RES = 1024
        w, h = init_image.size
        if max(w, h) > MAX_RES:
            ratio = MAX_RES / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            print(f"📏 Resizing uploaded image {w}x{h} → {new_size}")
            init_image = init_image.resize(new_size, Image.Resampling.LANCZOS)

        # Light upscale (for highres fix)
        upscale = 1.2
        new_size = (int(init_image.width * upscale), int(init_image.height * upscale))
        init_image = init_image.resize(new_size, Image.Resampling.LANCZOS)

        print_vram()

        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.4,
            num_inference_steps=30,
            guidance_scale=3
        )
    else:
        print(f"🖋️ Running Text-to-Image for prompt: {prompt}")
        pipe = TEXT2IMG_PIPE

        result = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=3
        )

    image = result.images[0]
    image.save(save_path)
    print(f"✅ Generated: {save_path}")

    print_vram()
    return save_path


if __name__ == "__main__":
    print("✨ Model loaded and ready.")
    path = generate_image("a realistic photo of an astronaut riding a horse on Mars, cinematic lighting")
    print(f"📁 Saved to: {path}")