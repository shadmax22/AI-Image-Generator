from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch
import random
import os

MODEL_PATH = "./models/model.safetensors"

print("🔄 Loading Stable Diffusion XL pipelines...")

text2img_pipe = StableDiffusionXLPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=torch.float16
).to("mps")

img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=torch.float32
).to("mps")

text2img_pipe.enable_attention_slicing()
text2img_pipe.enable_vae_slicing()
img2img_pipe.enable_attention_slicing()
img2img_pipe.enable_vae_slicing()

print("✅ Both pipelines loaded successfully!")


def generate_image(prompt: str, init_image: Image.Image | None = None) -> str:
    """Generate an image from prompt or prompt + image."""
    save_dir = "static"
    os.makedirs(save_dir, exist_ok=True)
    img_name = f"{random.randint(1000, 9999)}.png"
    save_path = os.path.join(save_dir, img_name)

    if init_image:
        print(f"🎨 Running Image-to-Image...")
        init_image = init_image.convert("RGB").resize((512, 512))
        image = img2img_pipe(
            prompt=prompt,
            image=init_image,
            strength=0.5,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
    else:
        print(f"🖋️ Running Text-to-Image for prompt: {prompt}")
        image = text2img_pipe(
            prompt,
            num_inference_steps=15,
            guidance_scale=7.5
        ).images[0]

    image.save(save_path)
    print(f"✅ Generated: {save_path}")
    return save_path