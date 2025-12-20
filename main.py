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

    Looks for the first significant peak in the autocorrelation,
    which indicates the repeating pattern size.
    """
    # Skip lag 0, look for first significant peak
    search_range = autocorr[min_size : max_size + 1]
    if len(search_range) == 0:
        return min_size

    # Find peaks by looking for local maxima
    peaks = []
    for i in range(1, len(search_range) - 1):
        if search_range[i] > search_range[i - 1] and search_range[i] > search_range[i + 1]:
            peaks.append((i + min_size, search_range[i]))

    if not peaks:
        # No clear peaks, return the maximum
        return int(np.argmax(search_range) + min_size)

    # Return the first strong peak (above a threshold relative to max)
    max_peak_val = max(p[1] for p in peaks)
    threshold = max_peak_val * 0.5

    for lag, val in peaks:
        if val >= threshold:
            return lag

    return peaks[0][0]


def detect_grid_size(image: np.ndarray, max_size: int = 50, square: bool = False) -> tuple[int, int]:
    """
    Detect the most likely original pixel grid size.

    Uses autocorrelation on rows and columns to find repeating patterns.

    Args:
        image: Input image array
        max_size: Maximum grid size to search for
        square: If True, return a uniform grid size

    Returns:
        (grid_width, grid_height) - the detected super-pixel size
    """
    # Convert to grayscale for analysis if color
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # Compute gradient to emphasize edges (pixel boundaries)
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))

    # Average gradients across all rows/columns
    avg_grad_x = np.mean(grad_x, axis=0)
    avg_grad_y = np.mean(grad_y, axis=1)

    # Compute autocorrelation
    autocorr_x = compute_autocorrelation_1d(avg_grad_x)
    autocorr_y = compute_autocorrelation_1d(avg_grad_y)

    # Find grid sizes
    grid_w = find_grid_size_1d(autocorr_x, max_size=max_size)
    grid_h = find_grid_size_1d(autocorr_y, max_size=max_size)

    if square:
        # Use the average, rounded to nearest integer
        grid_size = round((grid_w + grid_h) / 2)
        return grid_size, grid_size

    return grid_w, grid_h


def detect_grid_offset(image: np.ndarray, grid_w: int, grid_h: int, method: str = "center") -> tuple[int, int]:
    """
    Detect the grid offset (where the grid starts).

    For center sampling: finds offset where cell centers land on uniform color regions.
    For average/median: finds offset that minimizes variance within entire cells.

    Returns:
        (offset_x, offset_y) - the detected grid offset
    """
    best_offset_x = 0
    best_offset_y = 0
    min_score = float("inf")

    if method == "center":
        # For center sampling, we want the center of each cell to be in a uniform region
        # Sample a small area around each center and minimize its variance
        sample_radius = min(3, grid_w // 4, grid_h // 4)

        for off_x in range(grid_w):
            for off_y in range(grid_h):
                total_var = 0
                count = 0

                # Sample cells (skip edges)
                for cy in range(1, (image.shape[0] - off_y) // grid_h - 1, max(1, (image.shape[0] // grid_h) // 10)):
                    for cx in range(1, (image.shape[1] - off_x) // grid_w - 1, max(1, (image.shape[1] // grid_w) // 10)):
                        y_center = off_y + cy * grid_h + grid_h // 2
                        x_center = off_x + cx * grid_w + grid_w // 2

                        # Sample region around center
                        y_start = max(0, y_center - sample_radius)
                        y_end = min(image.shape[0], y_center + sample_radius + 1)
                        x_start = max(0, x_center - sample_radius)
                        x_end = min(image.shape[1], x_center + sample_radius + 1)

                        region = image[y_start:y_end, x_start:x_end]
                        if region.size > 0:
                            if len(image.shape) == 3:
                                var = np.mean([np.var(region[:, :, c]) for c in range(image.shape[2])])
                            else:
                                var = np.var(region)
                            total_var += var
                            count += 1

                if count > 0:
                    score = total_var / count
                    if score < min_score:
                        min_score = score
                        best_offset_x = off_x
                        best_offset_y = off_y
    else:
        # For average/median, minimize variance within entire cells
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        for off_x in range(grid_w):
            for off_y in range(grid_h):
                total_variance = 0
                count = 0

                for cy in range(0, (gray.shape[0] - off_y) // grid_h, max(1, gray.shape[0] // grid_h // 10)):
                    for cx in range(0, (gray.shape[1] - off_x) // grid_w, max(1, gray.shape[1] // grid_w // 10)):
                        y_start = off_y + cy * grid_h
                        y_end = min(y_start + grid_h, gray.shape[0])
                        x_start = off_x + cx * grid_w
                        x_end = min(x_start + grid_w, gray.shape[1])

                        cell = gray[y_start:y_end, x_start:x_end]
                        if cell.size > 0:
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
    image: np.ndarray, grid_w: int, grid_h: int, offset_x: int, offset_y: int,
    method: str = "center"
) -> np.ndarray:
    """
    Downsample the image by sampling each grid cell.

    Args:
        method: "center" (sample center pixel), "average" (mean of cell),
                "median" (median of cell)

    Returns:
        The downsampled image as a numpy array
    """
    # Calculate output dimensions
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

            if method == "center":
                # Sample the center pixel of the cell
                cy = grid_h // 2
                cx = grid_w // 2
                if len(image.shape) == 3:
                    output[y, x, :] = cell[cy, cx, :]
                else:
                    output[y, x] = cell[cy, cx]
            elif method == "median":
                if len(image.shape) == 3:
                    for c in range(channels):
                        output[y, x, c] = int(np.median(cell[:, :, c]))
                else:
                    output[y, x] = int(np.median(cell))
            else:  # average
                if len(image.shape) == 3:
                    for c in range(channels):
                        output[y, x, c] = int(np.round(np.mean(cell[:, :, c])))
                else:
                    output[y, x] = int(np.round(np.mean(cell)))

    return output


def analyze_image(image_path: Path, max_grid_size: int = 50, square: bool = False) -> dict:
    """
    Analyze an image and detect its original grid parameters.

    Returns:
        Dictionary with detected parameters
    """
    img = Image.open(image_path)
    img_array = np.array(img)

    grid_w, grid_h = detect_grid_size(img_array, max_size=max_grid_size, square=square)
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


def process_image(
    image_path: Path,
    output_path: Path | None = None,
    grid_size: tuple[int, int] | None = None,
    offset: tuple[int, int] | None = None,
    max_grid_size: int = 50,
    method: str = "center",
    square: bool = False,
) -> Path:
    """
    Process an image: detect grid and downsample.

    Args:
        image_path: Path to input image
        output_path: Path for output image (default: input_downsampled.png)
        grid_size: Override detected grid size (width, height)
        offset: Override detected offset (x, y)
        max_grid_size: Maximum grid size to search for
        square: Assume square grid (uniform scaling)

    Returns:
        Path to the output image
    """
    img = Image.open(image_path)
    img_array = np.array(img)

    # Detect or use provided grid parameters
    if grid_size is None:
        grid_w, grid_h = detect_grid_size(img_array, max_size=max_grid_size, square=square)
    else:
        grid_w, grid_h = grid_size

    if offset is None:
        offset_x, offset_y = detect_grid_offset(img_array, grid_w, grid_h, method=method)
    else:
        offset_x, offset_y = offset

    # Downsample
    result = downsample_image(img_array, grid_w, grid_h, offset_x, offset_y, method=method)

    # Save result
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_downsampled.png"

    result_img = Image.fromarray(result)
    result_img.save(output_path)

    return output_path


def upscale_image(image_path: Path, scale: int, output_path: Path | None = None) -> Path:
    """
    Upscale an image using nearest-neighbor interpolation (pixel-perfect).

    Args:
        image_path: Path to input image
        scale: Integer scale factor
        output_path: Path for output image (default: input_Nx.png)

    Returns:
        Path to the output image
    """
    img = Image.open(image_path)

    new_width = img.width * scale
    new_height = img.height * scale

    # Use NEAREST to preserve sharp pixel edges
    result = img.resize((new_width, new_height), Image.Resampling.NEAREST)

    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_{scale}x.png"

    result.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Detect and reverse poorly upscaled pixel art images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze image.png          # Detect grid size and offset
  %(prog)s process image.png          # Downsample to detected original size
  %(prog)s process image.png -o out.png --grid 4 4  # Force 4x4 grid
  %(prog)s upscale image.png 4        # Upscale 4x with nearest-neighbor
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze image and detect grid parameters")
    analyze_parser.add_argument("image", type=Path, help="Input image path")
    analyze_parser.add_argument(
        "--max-grid", type=int, default=50, help="Maximum grid size to search for (default: 50)"
    )
    analyze_parser.add_argument(
        "--square", action="store_true", help="Assume square grid (uniform scaling)"
    )

    # Process command
    process_parser = subparsers.add_parser("process", help="Downsample image to original resolution")
    process_parser.add_argument("image", type=Path, help="Input image path")
    process_parser.add_argument("-o", "--output", type=Path, help="Output image path")
    process_parser.add_argument(
        "--grid", type=int, nargs=2, metavar=("W", "H"), help="Override grid size (width height)"
    )
    process_parser.add_argument(
        "--offset", type=int, nargs=2, metavar=("X", "Y"), help="Override grid offset (x y)"
    )
    process_parser.add_argument(
        "--max-grid", type=int, default=50, help="Maximum grid size to search for (default: 50)"
    )
    process_parser.add_argument(
        "--method", choices=["center", "average", "median"], default="center",
        help="Sampling method: center (default), average, or median"
    )
    process_parser.add_argument(
        "--square", action="store_true", help="Assume square grid (uniform scaling)"
    )

    # Upscale command
    upscale_parser = subparsers.add_parser("upscale", help="Upscale image with nearest-neighbor (pixel-perfect)")
    upscale_parser.add_argument("image", type=Path, help="Input image path")
    upscale_parser.add_argument("scale", type=int, help="Integer scale factor (e.g., 2, 4, 8)")
    upscale_parser.add_argument("-o", "--output", type=Path, help="Output image path")

    args = parser.parse_args()

    if args.command == "analyze":
        if not args.image.exists():
            print(f"Error: Image not found: {args.image}")
            return 1

        result = analyze_image(args.image, max_grid_size=args.max_grid, square=args.square)

        print(f"Image: {args.image}")
        print(f"Current size: {result['current_size'][0]} x {result['current_size'][1]}")
        print(f"Detected grid size: {result['grid_size'][0]} x {result['grid_size'][1]}")
        print(f"Detected offset: ({result['offset'][0]}, {result['offset'][1]})")
        print(f"Detected original size: {result['detected_original_size'][0]} x {result['detected_original_size'][1]}")

    elif args.command == "process":
        if not args.image.exists():
            print(f"Error: Image not found: {args.image}")
            return 1

        grid_size = tuple(args.grid) if args.grid else None
        offset = tuple(args.offset) if args.offset else None

        # First analyze
        result = analyze_image(args.image, max_grid_size=args.max_grid, square=args.square)

        print(f"Input: {args.image} ({result['current_size'][0]} x {result['current_size'][1]})")

        if grid_size:
            print(f"Using grid size: {grid_size[0]} x {grid_size[1]} (override)")
        else:
            print(f"Detected grid size: {result['grid_size'][0]} x {result['grid_size'][1]}")

        if offset:
            print(f"Using offset: ({offset[0]}, {offset[1]}) (override)")
        else:
            print(f"Detected offset: ({result['offset'][0]}, {result['offset'][1]})")

        output_path = process_image(
            args.image,
            output_path=args.output,
            grid_size=grid_size,
            offset=offset,
            max_grid_size=args.max_grid,
            method=args.method,
            square=args.square,
        )

        # Get output dimensions
        out_img = Image.open(output_path)
        print(f"Output: {output_path} ({out_img.width} x {out_img.height})")

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
