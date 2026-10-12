//! The real MediaPipe models on a real picture, when both are available (they are downloaded, never
//! committed): `FILMCRAFT_FACE_MODELS=<dir>` with `face_detection_short_range.tflite` and
//! `face_landmark.tflite`, and `FILMCRAFT_FACE_IMAGE=<png>` showing one face. Skipped otherwise.

use filmcraft_vision::{DetectorKind, FaceModels, Image};

fn setup() -> Option<(FaceModels, image::RgbaImage)> {
    let dir = std::path::PathBuf::from(std::env::var_os("FILMCRAFT_FACE_MODELS")?);
    let img = image::open(std::env::var_os("FILMCRAFT_FACE_IMAGE")?).ok()?.to_rgba8();
    let det = std::fs::read(dir.join("face_detection_short_range.tflite")).ok()?;
    let mesh = std::fs::read(dir.join("face_landmark.tflite")).ok()?;
    Some((FaceModels::load(&det, DetectorKind::ShortRange, Some(&mesh)).unwrap(), img))
}

#[test]
fn eyes_follow_shifts_and_mirroring() {
    let Some((m, img)) = setup() else {
        eprintln!("skipped: set FILMCRAFT_FACE_MODELS and FILMCRAFT_FACE_IMAGE");
        return;
    };
    let (w, h) = (img.width() as usize, img.height() as usize);
    let faces = m.detect(&Image::new(w, h, img.as_raw()).unwrap(), &Default::default()).unwrap();
    let f = faces.first().expect("a face");
    assert_eq!(f.method, filmcraft_vision::Method::Mesh);
    assert!(f.interocular() > 8.0);

    // precision scales with the face: ±1 px on a 4K talking head (interocular ~210 px), ±1.6 px on
    // a webcam inset (~40 px)
    let tol = (f.interocular() * 0.04).max(1.0);
    // shift the picture 37 px right and 23 px down: the eyes move as much
    let (dx, dy) = (37usize, 23usize);
    let mut shifted = image::RgbaImage::new(img.width(), img.height());
    image::imageops::replace(&mut shifted, &img, dx as i64, dy as i64);
    let g = m.detect(&Image::new(w, h, shifted.as_raw()).unwrap(), &Default::default()).unwrap();
    let g = g.first().expect("the shifted face");
    let (a, b) = (f.eye_mid(), g.eye_mid());
    assert!((b.x - a.x - dx as f64).abs() < tol && (b.y - a.y - dy as f64).abs() < tol, "{a:?} → {b:?}");

    // mirror: same eye height and mirrored x (±2 px: the model is not exactly symmetric; ~1 px at 4K)
    let flipped = image::imageops::flip_horizontal(&img);
    let k = m.detect(&Image::new(w, h, flipped.as_raw()).unwrap(), &Default::default()).unwrap();
    let k = k.first().expect("the mirrored face");
    assert!((k.eye_mid().y - a.y).abs() < 2.0 * tol, "{a:?} vs {:?}", k.eye_mid());
    assert!((k.eye_mid().x - (w as f64 - a.x)).abs() < 2.0 * tol, "{a:?} vs {:?}", k.eye_mid());
}

#[test]
fn hostile_pictures_find_nothing() {
    let Some((m, _)) = setup() else { return };
    for (w, h, v) in [(1usize, 1usize, 0u8), (7, 3000, 255), (640, 360, 0), (640, 360, 128)] {
        let px = vec![v; w * h * 4];
        let faces = m.detect(&Image::new(w, h, &px).unwrap(), &Default::default()).unwrap();
        assert!(faces.is_empty());
    }
}
