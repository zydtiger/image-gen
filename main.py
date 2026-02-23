"""Image generation script using Z-Image-Turbo."""

import torch
from diffusers import ZImageImg2ImgPipeline
from PIL import Image

# Load the Img2Img pipeline
pipe = ZImageImg2ImgPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    variant="bf16",
    revision="refs/pr/102"
)
pipe.to("cuda")

# Create a neutral starting image (white 1024x1024)
init_image = Image.new("RGB", (1024, 1024), (255, 255, 255))

# Prompt for generation
prompt = "1girl"

# Generate based on the starting image
image = pipe(
    prompt=prompt,
    image=init_image,
    strength=1.0,  # Full transformation from the starting image
    num_inference_steps=9,
    guidance_scale=0.0,
).images[0]

# Save the generated image
output_path = "outputs/generated.png"
image.save(output_path)
print(f"Image saved to {output_path}")