#!/usr/bin/env python3
"""
Pixel Downsizer - Detect and reverse poorly upscaled pixel art images.

Analyzes an image to find the underlying pixel grid from a poorly upscaled
low-resolution image, then downsamples it back to the original resolution.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def find_grid_size_from_gradients(
    gradients: np.ndarray, min_size: int = 2, max_size: int = 50
) -> int:
    """
    Find grid size by analyzing spacing between gradient peaks.

    Averages all rows/columns into a single signal, finds peaks,
    and returns the median spacing between consecutive peaks.
    """
    # Average across all rows/columns to get a 1D signal
    if gradients.ndim == 2:
        signal = np.mean(gradients, axis=0)
    else:
        signal = gradients

    # Find threshold for peaks (use high percentile to get strong edges)
    threshold = np.percentile(signal, 90)

    # Find peak positions
    peaks = []
    for i in range(1, len(signal) - 1):
        if (
            signal[i] > threshold
            and signal[i] > signal[i - 1]
            and signal[i] > signal[i + 1]
        ):
            peaks.append(i)

    if len(peaks) < 2:
        return min_size

    # Compute spacing between consecutive peaks
    spacings = np.diff(peaks)

    # Filter spacings within valid range
    valid_spacings = spacings[(spacings >= min_size) & (spacings <= max_size)]

    if len(valid_spacings) == 0:
        return min_size

    # Return the median spacing (robust to outliers)
    return int(np.median(valid_spacings))


def detect_offset(image: np.ndarray, grid_w: int, grid_h: int) -> tuple[int, int]:
    """
    Detect the grid offset by finding the alignment with minimum variance.

    The correct offset will align grid cells with uniform regions, resulting
    in lower variance within each cell.

    Args:
        image: Input image array
        grid_w: Grid width
        grid_h: Grid height

    Returns:
        (offset_x, offset_y) - the detected grid offset
    """
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    best_offset = (0, 0)
    min_variance = float("inf")

    # Try all possible offsets
    for offset_y in range(grid_h):
        for offset_x in range(grid_w):
            # Crop and trim to grid
            cropped = gray[offset_y:, offset_x:]
            out_h = cropped.shape[0] // grid_h
            out_w = cropped.shape[1] // grid_w

            if out_h < 2 or out_w < 2:
                continue

            cropped = cropped[: out_h * grid_h, : out_w * grid_w]

            # Reshape into grid cells
            reshaped = cropped.reshape(out_h, grid_h, out_w, grid_w)

            # Compute variance within each cell and take the mean
            cell_variances = np.var(reshaped, axis=(1, 3))
            mean_variance = np.mean(cell_variances)

            if mean_variance < min_variance:
                min_variance = mean_variance
                best_offset = (offset_x, offset_y)

    return best_offset


def detect_grid_size(image: np.ndarray, max_size: int = 50) -> tuple[int, int]:
    """
    Detect the most likely original pixel grid size.

    Uses peak spacing in gradients to find repeating patterns.

    Args:
        image: Input image array
        max_size: Maximum grid size to search for

    Returns:
        (grid_width, grid_height) - the detected super-pixel size
    """
    # Convert to grayscale for analysis if color
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # Compute gradients to find pixel boundaries
    grad_x = np.abs(np.diff(gray, axis=1))  # horizontal edges, shape (H, W-1)
    grad_y = np.abs(np.diff(gray, axis=0))  # vertical edges, shape (H-1, W)

    # Find grid sizes from gradient peak spacing
    grid_w = find_grid_size_from_gradients(grad_x, max_size=max_size)
    grid_h = find_grid_size_from_gradients(grad_y.T, max_size=max_size)

    return grid_w, grid_h


def downsample_image(
    image: np.ndarray, grid_w: int, grid_h: int, offset_x: int, offset_y: int
) -> np.ndarray:
    """
    Downsample the image by taking the average of each grid cell.

    Returns:
        The downsampled image as a numpy array
    """
    # Crop to align with grid
    cropped = image[offset_y:, offset_x:]

    out_h = cropped.shape[0] // grid_h
    out_w = cropped.shape[1] // grid_w

    # Trim to exact multiple of grid
    cropped = cropped[: out_h * grid_h, : out_w * grid_w]

    if len(image.shape) == 3:
        channels = image.shape[2]
        # Reshape to group pixels into grid cells: (out_h, grid_h, out_w, grid_w, channels)
        reshaped = cropped.reshape(out_h, grid_h, out_w, grid_w, channels)
        # Take mean over grid_h and grid_w axes (axes 1 and 3)
        output = np.median(reshaped, axis=(1, 3)).astype(np.uint8)
    else:
        # Reshape to group pixels into grid cells: (out_h, grid_h, out_w, grid_w)
        reshaped = cropped.reshape(out_h, grid_h, out_w, grid_w)
        # Take mean over grid_h and grid_w axes (axes 1 and 3)
        output = np.median(reshaped, axis=(1, 3)).astype(np.uint8)

    return output


def upscale_image(image: Image.Image, scale: int) -> Image.Image:
    """
    Upscale an image using nearest-neighbor interpolation.

    Args:
        image: Input PIL Image
        scale: Integer scale factor

    Returns:
        Upscaled PIL Image
    """
    new_w = image.width * scale
    new_h = image.height * scale
    return image.resize((new_w, new_h), Image.Resampling.NEAREST)


def main():
    parser = argparse.ArgumentParser(
        description="Detect and reverse poorly upscaled pixel art images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png
  %(prog)s image.png --scale 4
        """,
    )

    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "--scale",
        type=int,
        help="Scale factor for output (default: 1x for downsampled only)",
    )

    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    # Load image
    img = Image.open(args.image)
    img_array = np.array(img)

    # Detect grid and offset
    grid_w, grid_h = detect_grid_size(img_array)
    offset_x, offset_y = detect_offset(img_array, grid_w, grid_h)

    print(f"Input: {args.image} ({img.width} x {img.height})")
    print(f"Detected grid size: {grid_w} x {grid_h}")
    print(f"Detected offset: ({offset_x}, {offset_y})")

    # Downsample
    downsampled = downsample_image(img_array, grid_w, grid_h, offset_x, offset_y)
    downsampled_img = Image.fromarray(downsampled)

    print(f"Downsampled: ({downsampled_img.width} x {downsampled_img.height})")

    # Generate output filename and upscale
    scale_factor = args.scale if args.scale is not None else 1
    output_path = args.image.parent / f"{args.image.stem}_{scale_factor}x.png"

    output_img = upscale_image(downsampled_img, scale_factor)
    output_img.save(output_path)

    print(f"Output: {output_path} ({output_img.width} x {output_img.height})")

    return 0


if __name__ == "__main__":
    exit(main())
