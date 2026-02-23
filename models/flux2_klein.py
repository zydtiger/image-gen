import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype)
pipe.enable_model_cpu_offload()

prompt = "1girl"

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=5,
    generator=torch.Generator(device=device).manual_seed(0)
).images[0] # type: ignore

output_path = "outputs/generated.png"
image.save(output_path)
print(f"Image saved to {output_path}")