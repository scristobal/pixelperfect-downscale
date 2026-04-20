//! Detect and reverse poorly upscaled pixel art images.
//!
//! Uses spectral analysis of pixel differences to find the repeating grid
//! period, then downsamples by taking the median colour per cell.

use std::collections::HashMap;

use image::{ImageBuffer, Rgb, RgbImage};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// Result of grid detection on an image.
pub struct GridInfo {
    pub grid_w: usize,
    pub grid_h: usize,
    pub offset_x: usize,
    pub offset_y: usize,
    /// Fraction of pixel variance explained by cell-median quantization
    /// (0..=1). Near 1 for a real upscaled grid, near 0 when the image is
    /// already at native resolution.
    pub confidence: f64,
}

/// Average autocorrelation at the first harmonics (1x..5x) of `period`.
fn harmonic_score(period: usize, autocorr: &[f64], max_lag: usize) -> f64 {
    let mut score = 0.0;
    let mut count = 0;
    let mut lag = period;
    while lag < max_lag && count < 5 {
        score += autocorr[lag];
        count += 1;
        lag += period;
    }
    if count > 0 { score / count as f64 } else { 0.0 }
}

/// Find the dominant period in a 1-D signal via autocorrelation.
///
/// Candidates are local maxima in the autocorrelation. Each candidate and
/// its +/-1 neighbours are scored by their harmonic average to handle
/// compression-smeared peaks.
fn detect_period(signal: &[f64], min_size: usize, max_size: usize) -> usize {
    let n = signal.len();
    let fft_len = n.next_power_of_two();

    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(fft_len);
    let fft_inv = planner.plan_fft_inverse(fft_len);

    let mut buf: Vec<Complex<f64>> = signal
        .iter()
        .map(|&v| Complex::new(v, 0.0))
        .chain(std::iter::repeat_n(Complex::new(0.0, 0.0), fft_len - n))
        .collect();

    fft_fwd.process(&mut buf);
    for c in buf.iter_mut() {
        *c = Complex::new(c.norm_sqr(), 0.0);
    }
    fft_inv.process(&mut buf);

    let scale = 1.0 / buf[0].re.max(1e-30);
    let autocorr: Vec<f64> = buf.iter().map(|c| c.re * scale).collect();

    let max_lag = max_size.min(n / 2);

    // Collect local maxima in the candidate range.
    let mut peaks: Vec<usize> = Vec::new();
    for lag in min_size..max_lag {
        if autocorr[lag] > autocorr[lag - 1] && autocorr[lag] >= autocorr[lag + 1] {
            peaks.push(lag);
        }
    }

    if peaks.is_empty() {
        return min_size;
    }

    // Score each peak and its +/-1 neighbours by harmonic average.
    let half = n / 2;
    let mut best_lag = peaks[0];
    let mut best_hscore = f64::NEG_INFINITY;
    for &lag in &peaks {
        for offset in [0isize, -1, 1] {
            let candidate = lag as isize + offset;
            if candidate < min_size as isize || candidate >= max_lag as isize {
                continue;
            }
            let candidate = candidate as usize;
            let hs = harmonic_score(candidate, &autocorr, half);
            if hs > best_hscore {
                best_hscore = hs;
                best_lag = candidate;
            }
        }
    }

    best_lag
}

/// Fraction of pixel variance explained by quantizing each cell to its
/// per-channel median. ~1 when the grid is real (cells near-uniform), ~0
/// when the image is already at native resolution.
fn explained_variance(
    img: &RgbImage,
    grid_w: usize,
    grid_h: usize,
    offset_x: usize,
    offset_y: usize,
) -> f64 {
    let (width, height) = img.dimensions();
    let (w, h) = (width as usize, height as usize);
    let n_cells_x = w.saturating_sub(offset_x) / grid_w;
    let n_cells_y = h.saturating_sub(offset_y) / grid_h;
    if n_cells_x == 0 || n_cells_y == 0 {
        return 0.0;
    }
    let covered_w = n_cells_x * grid_w;
    let covered_h = n_cells_y * grid_h;
    let total_px = (covered_w * covered_h) as f64;

    let mut sums = [0.0_f64; 3];
    for y in 0..covered_h {
        for x in 0..covered_w {
            let p = img
                .get_pixel((offset_x + x) as u32, (offset_y + y) as u32)
                .0;
            for c in 0..3 {
                sums[c] += p[c] as f64;
            }
        }
    }
    let means = [sums[0] / total_px, sums[1] / total_px, sums[2] / total_px];

    let mut total_ss = 0.0_f64;
    let mut within_ss = 0.0_f64;
    let cell_px = grid_w * grid_h;
    let mut chs: [Vec<u8>; 3] = [
        Vec::with_capacity(cell_px),
        Vec::with_capacity(cell_px),
        Vec::with_capacity(cell_px),
    ];
    for by in 0..n_cells_y {
        for bx in 0..n_cells_x {
            for c in 0..3 {
                chs[c].clear();
            }
            for dy in 0..grid_h {
                for dx in 0..grid_w {
                    let p = img
                        .get_pixel(
                            (offset_x + bx * grid_w + dx) as u32,
                            (offset_y + by * grid_h + dy) as u32,
                        )
                        .0;
                    for c in 0..3 {
                        chs[c].push(p[c]);
                    }
                }
            }
            let mut median = [0.0_f64; 3];
            for c in 0..3 {
                chs[c].sort_unstable();
                median[c] = chs[c][chs[c].len() / 2] as f64;
            }
            for c in 0..3 {
                for &v in &chs[c] {
                    let dw = v as f64 - median[c];
                    within_ss += dw * dw;
                    let dt = v as f64 - means[c];
                    total_ss += dt * dt;
                }
            }
        }
    }

    if total_ss <= 1e-10 {
        return 1.0;
    }
    1.0 - (within_ss / total_ss)
}

/// Find the pixel offset where grid cells start.
///
/// `h_diff[x]` holds the transition between pixel x and x+1. If cells start
/// at pixel `off` with period `grid`, boundaries land at `off + k*grid - 1`
/// for k >= 1, so the offset whose boundary positions best line up with the
/// diff peaks is the cell start itself.
fn detect_offset(diff_signal: &[f64], grid_size: usize) -> usize {
    let n = diff_signal.len();
    let mut best_offset = 0;
    let mut best_score = -1.0_f64;

    for off in 0..grid_size {
        let mut score = 0.0;
        let mut pos = off + grid_size - 1;
        while pos < n {
            score += diff_signal[pos];
            pos += grid_size;
        }
        if score > best_score {
            best_score = score;
            best_offset = off;
        }
    }

    best_offset
}

/// Detect the pixel grid by finding spectral peaks in pixel differences.
pub fn detect_grid(img: &RgbImage, min_size: usize, max_size: usize) -> GridInfo {
    let (width, height) = img.dimensions();
    let (w, h) = (width as usize, height as usize);

    let max_size = max_size.min(h / 2).min(w / 2);
    if max_size < min_size {
        return GridInfo {
            grid_w: min_size,
            grid_h: min_size,
            offset_x: 0,
            offset_y: 0,
            confidence: 0.0,
        };
    }

    // Sum absolute pixel differences across all rows/columns and channels.
    let mut h_diff = vec![0.0_f64; w - 1];
    let mut v_diff = vec![0.0_f64; h - 1];

    for y in 0..h {
        for x in 0..(w - 1) {
            let p0 = img.get_pixel(x as u32, y as u32).0;
            let p1 = img.get_pixel((x + 1) as u32, y as u32).0;
            for c in 0..3 {
                h_diff[x] += (p0[c] as f64 - p1[c] as f64).abs();
            }
        }
    }

    for y in 0..(h - 1) {
        for x in 0..w {
            let p0 = img.get_pixel(x as u32, y as u32).0;
            let p1 = img.get_pixel(x as u32, (y + 1) as u32).0;
            for c in 0..3 {
                v_diff[y] += (p0[c] as f64 - p1[c] as f64).abs();
            }
        }
    }

    let grid_w = detect_period(&h_diff, min_size, max_size);
    let grid_h = detect_period(&v_diff, min_size, max_size);
    let offset_x = detect_offset(&h_diff, grid_w);
    let offset_y = detect_offset(&v_diff, grid_h);
    let confidence = explained_variance(img, grid_w, grid_h, offset_x, offset_y);

    GridInfo {
        grid_w,
        grid_h,
        offset_x,
        offset_y,
        confidence,
    }
}

/// Downsample by taking the median colour of each grid cell.
pub fn downsample_image(
    img: &RgbImage,
    grid_w: usize,
    grid_h: usize,
    offset_x: usize,
    offset_y: usize,
) -> RgbImage {
    let (width, height) = img.dimensions();
    let (w, h) = (width as usize, height as usize);

    let out_w = w.saturating_sub(offset_x) / grid_w;
    let out_h = h.saturating_sub(offset_y) / grid_h;
    let mut output = ImageBuffer::new(out_w as u32, out_h as u32);

    for by in 0..out_h {
        for bx in 0..out_w {
            let mut channels: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            for dy in 0..grid_h {
                for dx in 0..grid_w {
                    let px = (offset_x + bx * grid_w + dx) as u32;
                    let py = (offset_y + by * grid_h + dy) as u32;
                    let p = img.get_pixel(px, py).0;
                    for c in 0..3 {
                        channels[c].push(p[c]);
                    }
                }
            }
            let mut rgb = [0u8; 3];
            for c in 0..3 {
                channels[c].sort_unstable();
                rgb[c] = channels[c][channels[c].len() / 2];
            }
            output.put_pixel(bx as u32, by as u32, Rgb(rgb));
        }
    }

    output
}

/// Result of automatic palette detection.
pub struct PaletteInfo {
    pub unique_colors: usize,
    pub palette_size: usize,
}

/// Detect the dominant palette of an image and remap it to use only those
/// colours.
///
/// Colours are collected from `img` (expected to be the median-downsampled
/// output, which has already removed most compression noise). Clusters are
/// merged by weighted average until every pair has squared Euclidean
/// distance >= `MIN_DIST_SQ`.
pub fn clean_palette(img: &RgbImage) -> (RgbImage, PaletteInfo) {
    let (w, h) = img.dimensions();

    let mut freq: HashMap<[u8; 3], usize> = HashMap::new();
    for y in 0..h {
        for x in 0..w {
            *freq.entry(img.get_pixel(x, y).0).or_default() += 1;
        }
    }

    let unique_colors = freq.len();

    if unique_colors <= 2 {
        return (
            img.clone(),
            PaletteInfo {
                unique_colors,
                palette_size: unique_colors,
            },
        );
    }

    // Merge radius in squared RGB distance (~22 units). Wide enough to
    // absorb compression bleed and per-channel-median artefacts at cell
    // boundaries without merging distinct palette entries.
    const MERGE_DIST_SQ: f64 = 500.0;

    // Leader clustering with fixed seeds. Colours are processed in
    // descending frequency order: each either lands within MERGE_DIST_SQ of
    // an existing seed (absorbed into it) or becomes a new seed. Seeds are
    // never averaged, so by induction every pair is >= MERGE_DIST_SQ apart
    // — which makes a second pass over the cleaned output a no-op.
    let mut colors: Vec<([u8; 3], usize)> = freq.into_iter().collect();
    colors.sort_by(|a, b| b.1.cmp(&a.1));

    let mut seeds: Vec<[u8; 3]> = Vec::new();
    for (c, _) in colors {
        let mut near = false;
        for s in &seeds {
            let dr = c[0] as f64 - s[0] as f64;
            let dg = c[1] as f64 - s[1] as f64;
            let db = c[2] as f64 - s[2] as f64;
            if dr * dr + dg * dg + db * db < MERGE_DIST_SQ {
                near = true;
                break;
            }
        }
        if !near {
            seeds.push(c);
        }
    }

    let palette = seeds;
    let palette_size = palette.len();

    let mut output = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y).0;
            let nearest = palette
                .iter()
                .min_by_key(|c| {
                    let dr = p[0] as i32 - c[0] as i32;
                    let dg = p[1] as i32 - c[1] as i32;
                    let db = p[2] as i32 - c[2] as i32;
                    (dr * dr + dg * dg + db * db) as u32
                })
                .unwrap();
            output.put_pixel(x, y, Rgb(*nearest));
        }
    }

    (
        output,
        PaletteInfo {
            unique_colors,
            palette_size,
        },
    )
}

/// Upscale an image using nearest-neighbour interpolation.
pub fn upscale_image(img: &RgbImage, scale: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    let mut output = ImageBuffer::new(w * scale, h * scale);

    for y in 0..(h * scale) {
        for x in 0..(w * scale) {
            output.put_pixel(x, y, *img.get_pixel(x / scale, y / scale));
        }
    }

    output
}
