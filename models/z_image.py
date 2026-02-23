import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

# 1. Load the BASE model (Notice we changed the repository name)
# We still use bfloat16 to save VRAM and speed up generation
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image", 
    torch_dtype=torch.bfloat16
)

# Move the model to the GPU
pipe.to("cuda")

# 2. Define your Prompts
# Positive: What you want to see
prompt = "1anime girl"

# Negative: What you want the model to actively avoid drawing
negative_prompt = "blurry, low quality, deformed geometry, dark, gloomy, stormy, text, watermarks, westerner, realism, birds."

# 3. Generate the Image
# Notice the changed parameters compared to the Turbo version!
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt, # <-- Injecting the negative constraints
    num_inference_steps=40,          # <-- Base needs 30-50 steps to look good
    guidance_scale=4.5,              # <-- CFG must be high enough to enforce the negative prompt
    width=1024,
    height=1024
).images[0] # type: ignore

# 4. Save the result
output_path = "outputs/generated.png"
image.save(output_path)
print(f"Image saved to {output_path}")