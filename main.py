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


def compute_autocorrelation_1d(signal: np.ndarray) -> np.ndarray:
    """Compute normalized autocorrelation of a 1D signal."""
    n = len(signal)
    signal = signal - np.mean(signal)
    result = np.correlate(signal, signal, mode="full")
    result = result[n - 1 :]  # Take only positive lags
    if result[0] != 0:
        result = result / result[0]  # Normalize
    return result


def find_grid_size_1d(autocorr: np.ndarray, min_size: int = 2, max_size: int = 50) -> int:
    """
    Find the most likely grid size from autocorrelation.

    Returns the lag with maximum autocorrelation value.
    """
    search_range = autocorr[min_size : max_size + 1]
    if len(search_range) == 0:
        return min_size

    return int(np.argmax(search_range) + min_size)


def detect_grid_size(image: np.ndarray, max_size: int = 50) -> tuple[int, int]:
    """
    Detect the most likely original pixel grid size.

    Uses autocorrelation on concatenated row/column gradients to find repeating patterns.

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

    # Concatenate all rows for grid_w, all columns for grid_h
    concat_grad_x = grad_x.flatten()  # row by row
    concat_grad_y = grad_y.flatten(order='F')  # column by column

    # Compute autocorrelation
    autocorr_x = compute_autocorrelation_1d(concat_grad_x)
    autocorr_y = compute_autocorrelation_1d(concat_grad_y)

    # Find grid sizes
    grid_w = find_grid_size_1d(autocorr_x, max_size=max_size)
    grid_h = find_grid_size_1d(autocorr_y, max_size=max_size)

    return grid_w, grid_h


def detect_grid_offset(image: np.ndarray, grid_w: int, grid_h: int) -> tuple[int, int]:
    """
    Detect the grid offset (where the grid starts).

    Finds offset that minimizes mean variance within cells.

    Returns:
        (offset_x, offset_y) - the detected grid offset
    """
    best_offset_x = 0
    best_offset_y = 0
    min_score = float("inf")

    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    for off_x in range(grid_w):
        for off_y in range(grid_h):
            total_variance = 0
            count = 0

            for cy in range((gray.shape[0] - off_y) // grid_h):
                for cx in range((gray.shape[1] - off_x) // grid_w):
                    y_start = off_y + cy * grid_h
                    y_end = y_start + grid_h
                    x_start = off_x + cx * grid_w
                    x_end = x_start + grid_w

                    cell = gray[y_start:y_end, x_start:x_end]
                    total_variance += np.var(cell)
                    count += 1

            if count > 0:
                score = total_variance / count
                if score < min_score:
                    min_score = score
                    best_offset_x = off_x
                    best_offset_y = off_y

    return best_offset_x, best_offset_y


def downsample_image(
    image: np.ndarray, grid_w: int, grid_h: int, offset_x: int, offset_y: int
) -> np.ndarray:
    """
    Downsample the image by taking the median of each grid cell.

    Returns:
        The downsampled image as a numpy array
    """
    out_h = (image.shape[0] - offset_y) // grid_h
    out_w = (image.shape[1] - offset_x) // grid_w

    if len(image.shape) == 3:
        channels = image.shape[2]
        output = np.zeros((out_h, out_w, channels), dtype=np.uint8)
    else:
        output = np.zeros((out_h, out_w), dtype=np.uint8)

    for y in range(out_h):
        for x in range(out_w):
            y_start = offset_y + y * grid_h
            y_end = y_start + grid_h
            x_start = offset_x + x * grid_w
            x_end = x_start + grid_w

            cell = image[y_start:y_end, x_start:x_end]

            if len(image.shape) == 3:
                for c in range(channels):
                    output[y, x, c] = int(np.median(cell[:, :, c]))
            else:
                output[y, x] = int(np.median(cell))

    return output


def analyze_image(image_path: Path) -> dict:
    """
    Analyze an image and detect its original grid parameters.

    Returns:
        Dictionary with detected parameters
    """
    img = Image.open(image_path)
    img_array = np.array(img)

    grid_w, grid_h = detect_grid_size(img_array)
    offset_x, offset_y = detect_grid_offset(img_array, grid_w, grid_h)

    original_w = (img_array.shape[1] - offset_x) // grid_w
    original_h = (img_array.shape[0] - offset_y) // grid_h

    return {
        "current_size": (img.width, img.height),
        "grid_size": (grid_w, grid_h),
        "offset": (offset_x, offset_y),
        "detected_original_size": (original_w, original_h),
        "scale_factor": (grid_w, grid_h),
    }


def process_image(image_path: Path, output_path: Path) -> tuple[Path, Path]:
    """
    Process an image: detect grid, downsample, and re-upscale.

    Args:
        image_path: Path to input image
        output_path: Path for downsampled output image

    Returns:
        Tuple of (downsampled_path, reupscaled_path)
    """
    img = Image.open(image_path)
    img_array = np.array(img)

    grid_w, grid_h = detect_grid_size(img_array)
    offset_x, offset_y = detect_grid_offset(img_array, grid_w, grid_h)

    # Downsample
    result = downsample_image(img_array, grid_w, grid_h, offset_x, offset_y)

    # Save downsampled
    result_img = Image.fromarray(result)
    result_img.save(output_path)

    # Re-upscale with correct aspect ratio
    reupscaled_path = output_path.parent / f"{output_path.stem}_reupscaled.png"
    new_w = result_img.width * grid_w
    new_h = result_img.height * grid_h
    reupscaled_img = result_img.resize((new_w, new_h), Image.Resampling.NEAREST)
    reupscaled_img.save(reupscaled_path)

    return output_path, reupscaled_path


def upscale_image(image_path: Path, scale: int, output_path: Path) -> Path:
    """
    Upscale an image using nearest-neighbor interpolation (pixel-perfect).

    Args:
        image_path: Path to input image
        scale: Integer scale factor
        output_path: Path for output image

    Returns:
        Path to the output image
    """
    img = Image.open(image_path)

    new_width = img.width * scale
    new_height = img.height * scale

    result = img.resize((new_width, new_height), Image.Resampling.NEAREST)
    result.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Detect and reverse poorly upscaled pixel art images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze -i image.png
  %(prog)s process -i image.png -o output.png
  %(prog)s upscale -i image.png -o output.png --scale 4
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze image and detect grid parameters")
    analyze_parser.add_argument("-i", "--image", type=Path, required=True, help="Input image path")

    # Process command
    process_parser = subparsers.add_parser("process", help="Downsample image to original resolution")
    process_parser.add_argument("-i", "--image", type=Path, required=True, help="Input image path")
    process_parser.add_argument("-o", "--output", type=Path, required=True, help="Output image path")

    # Upscale command
    upscale_parser = subparsers.add_parser("upscale", help="Upscale image with nearest-neighbor (pixel-perfect)")
    upscale_parser.add_argument("-i", "--image", type=Path, required=True, help="Input image path")
    upscale_parser.add_argument("-o", "--output", type=Path, required=True, help="Output image path")
    upscale_parser.add_argument("--scale", type=int, default=1, help="Integer scale factor (default: 1)")

    args = parser.parse_args()

    if args.command == "analyze":
        if not args.image.exists():
            print(f"Error: Image not found: {args.image}")
            return 1

        result = analyze_image(args.image)

        print(f"Image: {args.image}")
        print(f"Current size: {result['current_size'][0]} x {result['current_size'][1]}")
        print(f"Detected grid size: {result['grid_size'][0]} x {result['grid_size'][1]}")
        print(f"Detected offset: ({result['offset'][0]}, {result['offset'][1]})")
        print(f"Detected original size: {result['detected_original_size'][0]} x {result['detected_original_size'][1]}")

    elif args.command == "process":
        if not args.image.exists():
            print(f"Error: Image not found: {args.image}")
            return 1

        result = analyze_image(args.image)

        print(f"Input: {args.image} ({result['current_size'][0]} x {result['current_size'][1]})")
        print(f"Detected grid size: {result['grid_size'][0]} x {result['grid_size'][1]}")
        print(f"Detected offset: ({result['offset'][0]}, {result['offset'][1]})")

        downsampled_path, reupscaled_path = process_image(args.image, args.output)

        down_img = Image.open(downsampled_path)
        reup_img = Image.open(reupscaled_path)
        print(f"Downsampled: {downsampled_path} ({down_img.width} x {down_img.height})")
        print(f"Reupscaled: {reupscaled_path} ({reup_img.width} x {reup_img.height})")

    elif args.command == "upscale":
        if not args.image.exists():
            print(f"Error: Image not found: {args.image}")
            return 1

        if args.scale < 1:
            print(f"Error: Scale must be at least 1, got {args.scale}")
            return 1

        img = Image.open(args.image)
        print(f"Input: {args.image} ({img.width} x {img.height})")

        output_path = upscale_image(args.image, args.scale, args.output)

        out_img = Image.open(output_path)
        print(f"Output: {output_path} ({out_img.width} x {out_img.height})")

    return 0


if __name__ == "__main__":
    exit(main())
