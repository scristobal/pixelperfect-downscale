use std::collections::HashSet;
use std::io::Cursor;

use image::{ImageFormat, ImageReader, RgbImage};
use wasm_bindgen::prelude::*;

use crate::{clean_palette, detect_grid, downsample_image, upscale_image};

#[wasm_bindgen]
pub struct ProcessResult {
    pub grid_w: u32,
    pub grid_h: u32,
    pub offset_x: u32,
    pub offset_y: u32,
    pub confidence: f64,
    pub upscaled_detected: bool,
    pub input_colors: u32,
    pub output_colors: u32,
    pub width: u32,
    pub height: u32,
    png: Vec<u8>,
}

fn count_unique(img: &RgbImage) -> u32 {
    let mut set: HashSet<[u8; 3]> = HashSet::new();
    for p in img.pixels() {
        set.insert(p.0);
    }
    set.len() as u32
}

#[wasm_bindgen]
impl ProcessResult {
    #[wasm_bindgen(getter)]
    pub fn png(&self) -> Vec<u8> {
        self.png.clone()
    }
}

#[wasm_bindgen]
pub fn process_image(
    bytes: &[u8],
    scale: u32,
    max_grid: u32,
    clean: bool,
) -> Result<ProcessResult, JsError> {
    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| JsError::new(&format!("read: {e}")))?
        .decode()
        .map_err(|e| JsError::new(&format!("decode: {e}")))?
        .to_rgb8();

    let input_colors = count_unique(&img);

    let grid = detect_grid(&img, 2, max_grid as usize);
    let upscaled_detected = grid.confidence >= 0.7;

    let mut out = if upscaled_detected {
        downsample_image(&img, grid.grid_w, grid.grid_h, grid.offset_x, grid.offset_y)
    } else {
        img
    };

    if clean {
        let (cleaned, _info) = clean_palette(&out);
        out = cleaned;
    }

    if scale > 1 {
        out = upscale_image(&out, scale);
    }

    let output_colors = count_unique(&out);
    let (width, height) = out.dimensions();
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(out)
        .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
        .map_err(|e| JsError::new(&format!("encode: {e}")))?;

    Ok(ProcessResult {
        grid_w: grid.grid_w as u32,
        grid_h: grid.grid_h as u32,
        offset_x: grid.offset_x as u32,
        offset_y: grid.offset_y as u32,
        confidence: grid.confidence,
        upscaled_detected,
        input_colors,
        output_colors,
        width,
        height,
        png,
    })
}
