//! `cargo run -p filmcraft-vision --example faces -- <models dir> [short|full] <picture.png>...`
//! Prints the faces found in each picture and writes `<picture>.faces.png` with the eye centres
//! and boxes marked (manual validation on real footage).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().ok_or("models dir")?);
    let kind = args.next().ok_or("short|full")?;
    let (file, kind) = match kind.as_str() {
        "short" => ("face_detection_short_range.tflite", filmcraft_vision::DetectorKind::ShortRange),
        _ => ("face_detection_full_range.tflite", filmcraft_vision::DetectorKind::FullRange),
    };
    let mesh = std::fs::read(dir.join("face_landmark.tflite"))?;
    let m = filmcraft_vision::FaceModels::load(&std::fs::read(dir.join(file))?, kind, Some(&mesh))?;
    for p in args {
        let mut img = image::open(&p)?.to_rgba8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let t0 = std::time::Instant::now();
        let faces = m.detect(&filmcraft_vision::Image::new(w, h, img.as_raw())?, &Default::default())?;
        println!("{p}: {} faces in {:?}", faces.len(), t0.elapsed());
        for f in &faces {
            println!(
                "  score {:.3} mesh {:?} {:?} box ({:.0},{:.0} {:.0}×{:.0}) L ({:.1},{:.1}) R ({:.1},{:.1}) mid y {:.1} iod {:.1} angle {:.2}°",
                f.score,
                f.mesh_score,
                f.method,
                f.bbox.x,
                f.bbox.y,
                f.bbox.w,
                f.bbox.h,
                f.left_eye.x,
                f.left_eye.y,
                f.right_eye.x,
                f.right_eye.y,
                f.eye_mid().y,
                f.interocular(),
                f.eye_line_deg()
            );
            let mut dot = |x: f64, y: f64, c: [u8; 4], r: i64| {
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (px, py) = (x as i64 + dx, y as i64 + dy);
                        if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                            img.put_pixel(px as u32, py as u32, image::Rgba(c));
                        }
                    }
                }
            };
            for q in &f.mesh {
                dot(q.x, q.y, [0, 255, 0, 255], 1);
            }
            dot(f.left_eye.x, f.left_eye.y, [255, 0, 0, 255], 3);
            dot(f.right_eye.x, f.right_eye.y, [255, 0, 0, 255], 3);
        }
        img.save(format!("{p}.faces.png"))?;
    }
    Ok(())
}
