# Pixel Downsizer

Detect and reverse poorly upscaled pixel art images.

## Installation

```bash
uv sync
```

## Usage

Downsample to detected original resolution:

```bash
uv run python main.py image.png
```

This outputs `image_1x.png` (downsampled).

Downsample and upscale with a specific scale factor:

```bash
uv run python main.py image.png --scale 4
```

This outputs `image_4x.png` (downsampled then upscaled 4x).

## Samples

| Original (upscaled) | Downsampled (1x) | Upscaled (8x) | Upscaled (28x) |
|---------------------|------------------|---------------|----------------|
| <a href="samples/sample.jpeg"><img src="samples/sample.jpeg" width="200"></a> | <a href="samples/sample_1x.png"><img src="samples/sample_1x.png" width="200"></a> | <a href="samples/sample_8x.png"><img src="samples/sample_8x.png" width="200"></a> | <a href="samples/sample_28x.png"><img src="samples/sample_28x.png" width="200"></a> |

Click on an image to see full resolution.
