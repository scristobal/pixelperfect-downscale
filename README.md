# Pixel Downsizer

Detect and reverse poorly upscaled pixel art images.

## Installation

```bash
uv sync
```

## Usage

### Analyze

Detect grid size and offset of an upscaled image:

```bash
uv run python main.py analyze -i image.png
```

### Process

Downsample to detected original resolution:

```bash
uv run python main.py process -i image.png -o output.png
```

### Upscale

Upscale with nearest-neighbor interpolation:

```bash
uv run python main.py upscale -i image.png -o output.png --scale 4
```

## Samples

| Original (upscaled) | Downsampled | Re-upscaled |
|---------------------|-------------|-------------|
| <a href="samples/sample.jpeg"><img src="samples/sample.jpeg" width="200"></a> | <a href="samples/downsampled.png"><img src="samples/downsampled.png" width="200"></a> | <a href="samples/downsampled_reupscaled.png"><img src="samples/downsampled_reupscaled.png" width="200"></a> |

Click on an image to see full resolution.
