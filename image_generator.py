from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import torch
import random
import os

# ---- CONFIG ----
BASE_MODEL = "./models/sd_xl_base_1.0.safetensors"
CUSTOM_MODEL_PATH = "./models/model.safetensors"  # your civitai model
SAVE_DIR = "static"

# ---- DEVICE SETUP ----
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"🔄 Loading Stable Diffusion XL base pipeline on device: {DEVICE}")

# ---- LOAD BASE COMPONENTS ----
tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder", torch_dtype=DTYPE)
vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=DTYPE)

# ---- LOAD CUSTOM UNET ----
unet = UNet2DConditionModel.from_single_file(
    CUSTOM_MODEL_PATH,
    torch_dtype=DTYPE
)

# ---- BUILD PIPELINES ----
text2img_pipe = StableDiffusionXLPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    torch_dtype=DTYPE
).to(DEVICE)

img2img_pipe = StableDiffusionXLImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    torch_dtype=DTYPE
).to(DEVICE)

# ---- OPTIMIZE ----
for pipe in [text2img_pipe, img2img_pipe]:
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

print("✅ Pipelines loaded successfully with custom CivitAI model!")


# ---- IMAGE GENERATION ----
def generate_image(prompt: str, init_image: Image.Image | None = None) -> str:
    """Generate an image from text or enhance an uploaded image."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    img_name = f"{random.randint(1000, 9999)}.png"
    save_path = os.path.join(SAVE_DIR, img_name)

    guidance_scale = 8
    num_inference_steps = 50

    if init_image:
        print("🎨 Running Image-to-Image...")
        init_image = init_image.convert("RGB")
        w, h = init_image.size

        if max(w, h) > 1024:
            factor = 1024 / max(w, h)
            init_image = init_image.resize((int(w * factor), int(h * factor)))

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