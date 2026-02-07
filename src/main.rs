use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use downscaler::{clean_palette, detect_grid, downsample_image, upscale_image};

#[derive(Parser)]
#[command(about = "Detect and reverse poorly upscaled pixel art images")]
struct Cli {
    /// Input image path
    image: PathBuf,

    /// Scale factor for output (default: 1x)
    #[arg(long)]
    scale: Option<u32>,

    /// Maximum grid cell size to search
    #[arg(long, default_value_t = 50)]
    max_grid: usize,

    /// Skip automatic palette cleaning
    #[arg(long)]
    no_clean: bool,
}

fn main() {
    let cli = Cli::parse();

    let img = image::open(&cli.image)
        .unwrap_or_else(|e| {
            eprintln!("Error: cannot open {}: {e}", cli.image.display());
            std::process::exit(1);
        })
        .to_rgb8();

    let (width, height) = img.dimensions();
    println!("Input: {} ({width} x {height})", cli.image.display());

    let t0 = Instant::now();
    let grid = detect_grid(&img, 2, cli.max_grid);
    let elapsed = t0.elapsed();

    let fmt_conf = |score: f64| {
        if score < 999.0 {
            format!("{score:.1}")
        } else {
            "high".to_string()
        }
    };
    println!(
        "Grid: {}x{} offset=({},{}) confidence=({},{}) [{:.3}s]",
        grid.grid_w,
        grid.grid_h,
        grid.offset_x,
        grid.offset_y,
        fmt_conf(grid.confidence_w),
        fmt_conf(grid.confidence_h),
        elapsed.as_secs_f64(),
    );
    if grid.confidence_w < 2.0 || grid.confidence_h < 2.0 {
        println!("Warning: low confidence — image may not be upscaled pixel art");
    }

    let mut downsampled =
        downsample_image(&img, grid.grid_w, grid.grid_h, grid.offset_x, grid.offset_y);
    let (dw, dh) = downsampled.dimensions();
    println!("Downsampled: {dw} x {dh}");

    if !cli.no_clean {
        let (cleaned, info) = clean_palette(&img, &downsampled);
        downsampled = cleaned;
        println!(
            "Palette: {} unique → {} dominant (removed {})",
            info.unique_colors,
            info.palette_size,
            info.unique_colors - info.palette_size,
        );
    }

    let scale_factor = cli.scale.unwrap_or(1);
    let stem = cli.image.file_stem().unwrap().to_string_lossy();
    let fallback = PathBuf::from(".");
    let parent = cli.image.parent().unwrap_or(&fallback);
    let output_path = parent.join(format!("{stem}_{scale_factor}x.png"));

    let output = if scale_factor > 1 {
        upscale_image(&downsampled, scale_factor)
    } else {
        downsampled
    };

    let (ow, oh) = output.dimensions();
    output.save(&output_path).unwrap_or_else(|e| {
        eprintln!("Error: cannot save {}: {e}", output_path.display());
        std::process::exit(1);
    });

    println!("Output: {} ({ow} x {oh})", output_path.display());
}
