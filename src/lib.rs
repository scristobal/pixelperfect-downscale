//! Detect and reverse poorly upscaled pixel art images.
//!
//! Uses spectral analysis of pixel differences to find the repeating grid
//! period, then downsamples by taking the median colour per cell.

use std::collections::HashMap;

use image::{ImageBuffer, Rgb, RgbImage};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Result of grid detection on an image.
pub struct GridInfo {
    pub grid_w: usize,
    pub grid_h: usize,
    pub offset_x: usize,
    pub offset_y: usize,
    pub confidence_w: f64,
    pub confidence_h: f64,
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
/// compression-smeared peaks. Confidence is the ratio of the winning peak
/// to the median autocorrelation over the search range.
fn detect_period(signal: &[f64], min_size: usize, max_size: usize) -> (usize, f64) {
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
        return (min_size, 0.0);
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

    // Confidence: how much the peak stands out from the background.
    let mut ac_slice: Vec<f64> = autocorr[min_size..max_lag].to_vec();
    ac_slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = ac_slice[ac_slice.len() / 2];
    let confidence = autocorr[best_lag] / median.max(1e-10);

    (best_lag, confidence)
}

/// Find the pixel offset where grid cells start.
///
/// The diff signal peaks at transitions *between* cells. We find the offset
/// that best aligns `off + k*grid - 1` with high diffs, then add 1 because
/// the cell starts on the pixel after the boundary.
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

    (best_offset + 1) % grid_size
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
            confidence_w: 0.0,
            confidence_h: 0.0,
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

    let (grid_w, confidence_w) = detect_period(&h_diff, min_size, max_size);
    let (grid_h, confidence_h) = detect_period(&v_diff, min_size, max_size);
    let offset_x = detect_offset(&h_diff, grid_w);
    let offset_y = detect_offset(&v_diff, grid_h);

    GridInfo {
        grid_w,
        grid_h,
        offset_x,
        offset_y,
        confidence_w,
        confidence_h,
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

/// Detect the dominant palette from a reference image and remap a target
/// image to use only those colours.
///
/// The reference image (typically the original upscaled input) provides a
/// clean frequency signal. Colours below 0.05% of total pixels are treated
/// as compression artifacts and discarded. Survivors are merged by weighted
/// average until all pairs have squared Euclidean distance >= 50.
pub fn clean_palette(reference: &RgbImage, target: &RgbImage) -> (RgbImage, PaletteInfo) {
    let (rw, rh) = reference.dimensions();
    let total_pixels = (rw as usize) * (rh as usize);

    let mut freq: HashMap<[u8; 3], usize> = HashMap::new();
    for y in 0..rh {
        for x in 0..rw {
            *freq.entry(reference.get_pixel(x, y).0).or_default() += 1;
        }
    }

    let unique_colors = freq.len();

    if unique_colors <= 2 {
        return (
            target.clone(),
            PaletteInfo {
                unique_colors,
                palette_size: unique_colors,
            },
        );
    }

    // Keep colours covering at least 0.05% of the reference image.
    let mut colors: Vec<([u8; 3], usize)> = freq.into_iter().collect();
    colors.sort_by(|a, b| b.1.cmp(&a.1));

    let threshold = (total_pixels as f64 * 0.0005).max(1.0) as usize;
    let candidate_count = colors
        .iter()
        .take_while(|&&(_, c)| c >= threshold)
        .count()
        .max(2);

    // Agglomerative merge until all pairs are perceptually distinct.
    const MIN_DIST_SQ: f64 = 50.0;

    let mut clusters: Vec<([f64; 3], usize)> = colors[..candidate_count]
        .iter()
        .map(|&(c, n)| ([c[0] as f64, c[1] as f64, c[2] as f64], n))
        .collect();

    loop {
        if clusters.len() <= 2 {
            break;
        }

        let mut best_dist = f64::MAX;
        let mut best_i = 0;
        let mut best_j = 1;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let dr = clusters[i].0[0] - clusters[j].0[0];
                let dg = clusters[i].0[1] - clusters[j].0[1];
                let db = clusters[i].0[2] - clusters[j].0[2];
                let d = dr * dr + dg * dg + db * db;
                if d < best_dist {
                    best_dist = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_dist >= MIN_DIST_SQ {
            break;
        }

        let (lo, hi) = if clusters[best_i].1 <= clusters[best_j].1 {
            (best_i, best_j)
        } else {
            (best_j, best_i)
        };
        let (ci, ni) = clusters[lo];
        let (cj, nj) = clusters[hi];
        let total = (ni + nj) as f64;
        let merged = [
            (ci[0] * ni as f64 + cj[0] * nj as f64) / total,
            (ci[1] * ni as f64 + cj[1] * nj as f64) / total,
            (ci[2] * ni as f64 + cj[2] * nj as f64) / total,
        ];
        clusters[hi] = (merged, ni + nj);
        clusters.swap_remove(lo);
    }

    let palette: Vec<[u8; 3]> = clusters
        .iter()
        .map(|(c, _)| [c[0].round() as u8, c[1].round() as u8, c[2].round() as u8])
        .collect();
    let palette_size = palette.len();

    // Remap every pixel in the target to its nearest palette entry.
    let (tw, th) = target.dimensions();
    let mut output = ImageBuffer::new(tw, th);
    for y in 0..th {
        for x in 0..tw {
            let p = target.get_pixel(x, y).0;
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
