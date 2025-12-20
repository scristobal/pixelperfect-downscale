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
uv run python main.py analyze image.png
```

### Process

Downsample to detected original resolution:

```bash
uv run python main.py process image.png
uv run python main.py process image.png -o output.png
uv run python main.py process image.png --grid 4 4
uv run python main.py process image.png --method average
```

Options:
- `-o, --output` - Output path
- `--grid W H` - Override detected grid size
- `--offset X Y` - Override detected offset
- `--method` - Sampling method: `center` (default), `average`, `median`
- `--square` - Assume uniform scaling

### Upscale

Upscale with nearest-neighbor interpolation:

```bash
uv run python main.py upscale image.png 4
```

## Samples

| Original (upscaled) | Downsampled | Re-upscaled |
|---------------------|-------------|-------------|
| <img src="samples/sample.jpeg" width="200"> | <img src="samples/downsampled_16x.png" width="200"> | <img src="samples/reupscaled.png" width="200"> |
