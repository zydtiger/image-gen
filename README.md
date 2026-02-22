# Image Generation with Z-Image-Turbo

A simple image generation project using Hugging Face's diffusers library with the
Z-Image-Turbo model.

## Requirements

- Python 3.12+
- CUDA-capable GPU (CUDA 13.0)
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zydtiger/image-gen.git
cd image-gen
```

2. Install dependencies using uv:
```bash
uv sync
```

## Usage

Run the image generation script:
```bash
uv run python generate.py
```

Generated images are saved to the `outputs/` directory.

## Configuration

The default settings in `generate.py`:
- **Model**: `Tongyi-MAI/Z-Image-Turbo`
- **Pipeline**: `ZImageImg2ImgPipeline`
- **Image Size**: 1024x1024
- **Default Prompt**: `"1girl"`
- **Inference Steps**: 9
- **Strength**: 1.0 (full transformation)

## Output

Generated images are saved as:
```
outputs/generated.png
```

## Dependencies

- `diffusers` - Hugging Face diffusion models
- `transformers` - Model utilities
- `accelerate` - Training and inference acceleration
- `torch` - PyTorch with CUDA 13.0 support
- `pillow` - Image processing