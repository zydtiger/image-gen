"""Image generation script using Z-Image-Turbo."""

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.z_image.pipeline_z_image_img2img import ZImageImg2ImgPipeline
from transformers import Qwen3Model
from PIL import Image

text_encoder = Qwen3Model.from_pretrained('Qwen/Qwen3-4B', torch_dtype=torch.bfloat16)

# Load the VAE from the base Z-Image model
vae = AutoencoderKL.from_pretrained("Tongyi-MAI/Z-Image", subfolder="vae", torch_dtype=torch.bfloat16)

# Load the Img2Img pipeline with the VAE component
pipe = ZImageImg2ImgPipeline.from_single_file(
    "./checkpoints/darkBeast.safetensors",
    torch_dtype=torch.bfloat16,
    text_encoder=text_encoder,
    vae=vae,
)
pipe.to("cuda")

# Create a neutral starting image (white 1024x1024)
init_image = Image.new("RGB", (1024, 1024), (255, 255, 255))

# Prompt for generation
prompt = """
1girl
"""

# Generate 10 images with different seeds
num_images = 10
base_seed = 42

for i in range(num_images):
    seed = base_seed + i
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Generate based on the starting image
    image = pipe(
        prompt=prompt,
        image=init_image,
        strength=1.0,  # Full transformation from the starting image
        num_inference_steps=15,
        guidance_scale=1.0,
        generator=generator,
    ).images[0] # type: ignore
    
    # Save the generated image with seed in filename
    output_path = f"outputs/generated_seed_{seed}.png"
    image.save(output_path)
    print(f"Image {i+1}/{num_images} saved to {output_path} (seed: {seed})")

print(f"\nGenerated {num_images} images successfully!")
