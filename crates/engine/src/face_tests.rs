//! `video.faces` / `video.alignEyes` / `video.auditJumpCuts` / `video.alignJumpCuts` with a
//! stand-in detector: synthetic "faces" are two red dots (the eyes) on grey stills, so the
//! geometry (source → sequence mapping, the Position solve, punch-in, border clamping, chaining
//! across cuts, verification on the rendered program, undo) is tested exactly without model files.
//! The real MediaPipe pipeline is tested in `crates/vision`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};

use crate::Session;
use crate::face::FaceDetector;
use crate::media_test_util::tmp_dir;
use filmcraft_vision::{BBox, DetectOptions, Face, Image, Method, Point};

/// Finds the red pixels, splits them at the widest horizontal gap into two eyes, and returns
/// their redness-weighted centroids (pixel centres at +0.5).
struct RedDots;

impl FaceDetector for RedDots {
    fn detect(&self, img: &Image, opts: &DetectOptions) -> Result<Vec<Face>, String> {
        let mut pts: Vec<(f64, f64, f64)> = Vec::new();
        for y in 0..img.height {
            for x in 0..img.width {
                let i = (y * img.width + x) * 4;
                let (r, g, b) = (img.rgba[i] as f64, img.rgba[i + 1] as f64, img.rgba[i + 2] as f64);
                let w = r - (g + b) / 2.0 - 100.0;
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let inside = opts.region.is_none_or(|q| px >= q.x && px < q.x + q.w && py >= q.y && py < q.y + q.h);
                if w > 0.0 && inside {
                    pts.push((px, py, w));
                }
            }
        }
        if pts.len() < 2 {
            return Ok(Vec::new());
        }
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let split = (1..pts.len()).max_by(|a, b| (pts[*a].0 - pts[*a - 1].0).total_cmp(&(pts[*b].0 - pts[*b - 1].0))).unwrap();
        let c = |v: &[(f64, f64, f64)]| {
            let w: f64 = v.iter().map(|p| p.2).sum();
            Point::new(v.iter().map(|p| p.0 * p.2).sum::<f64>() / w, v.iter().map(|p| p.1 * p.2).sum::<f64>() / w)
        };
        let (l, r) = (c(&pts[..split]), c(&pts[split..]));
        let d = r.x - l.x;
        Ok(vec![Face {
            score: 0.99,
            mesh_score: Some(1.0),
            bbox: BBox { x: l.x - d / 2.0, y: l.y - d, w: d * 2.0, h: d * 2.5 },
            left_eye: l,
            right_eye: r,
            method: Method::Mesh,
            mesh: Vec::new(),
        }])
    }
    fn id(&self) -> String {
        "red-dots".into()
    }
}

const W: u32 = 640;
const H: u32 = 360;

/// A grey still with two 5×5 red "eyes" centred on pixel (x, y) and (x + 40, y).
fn still(dir: &Path, name: &str, x: u32, y: u32) -> PathBuf {
    let mut img = image::RgbImage::from_pixel(W, H, image::Rgb([90, 90, 90]));
    for (cx, cy) in [(x, y), (x + 40, y)] {
        for dy in 0..5 {
            for dx in 0..5 {
                img.put_pixel(cx + dx - 2, cy + dy - 2, image::Rgb([255, 0, 0]));
            }
        }
    }
    let p = dir.join(name);
    img.save(&p).unwrap();
    p
}

/// A 30 fps sequence holding each still for one second on V1, back to back.
fn sequence_of(paths: &[PathBuf]) -> (Session, Vec<u64>) {
    use filmcraft_project::{SequenceSettings, TrackKind};
    use filmcraft_time::{FrameRate, Tick, TimeRange};
    let mut s = Session::default();
    let ps: Vec<String> = paths.iter().map(|p| p.to_string_lossy().into_owned()).collect();
    let r = s.execute("file.import", json!({"paths": ps})).unwrap();
    let items: Vec<filmcraft_project::ItemId> = r["items"].as_array().unwrap().iter().map(|v| filmcraft_project::ItemId(v.as_u64().unwrap())).collect();
    let rate = FrameRate::new(30, 1);
    let mut p = (*s.project).clone();
    let seq = p.new_sequence("Seq", SequenceSettings { width: W, height: H, frame_rate: rate, ..Default::default() }, 1, 1, None);
    let mut clips = Vec::new();
    for (k, it) in items.iter().enumerate() {
        let mut v =
            p.make_track_item(*it, TrackKind::Video, Tick::from_seconds_f64(k as f64), TimeRange::new(Tick::ZERO, Tick::from_seconds_f64(1.0)), rate).unwrap();
        for e in &mut v.effects {
            filmcraft_project::resolve_auto_points(e, (W, H), (W, H));
        }
        clips.push(v.id.0);
        p.sequence_mut(seq).unwrap().video_tracks[0].items.push(v);
    }
    s.project = Arc::new(p);
    s.state.active_sequence = Some(seq);
    s.state.open_sequences = vec![seq];
    s.face_detector = Some(Arc::new(RedDots));
    (s, clips)
}

fn setup(eyes: &[(u32, u32)]) -> (Session, Vec<u64>) {
    let dir = tmp_dir("face");
    let paths: Vec<PathBuf> = eyes.iter().enumerate().map(|(i, (x, y))| still(&dir, &format!("s{i}.png"), *x, *y)).collect();
    sequence_of(&paths)
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

#[test]
fn faces_reports_source_and_sequence_eyes() {
    let (mut s, c) = setup(&[(300, 150)]);
    let r = s.execute("video.faces", json!({"clip": c[0]})).unwrap();
    let face = &r["faces"][0];
    assert_eq!(r["source"], "clip");
    assert!((f(&face["source"]["eyeMid"]["x"]) - 320.5).abs() < 0.01, "{r}");
    assert!((f(&face["source"]["eyeMid"]["y"]) - 150.5).abs() < 0.01, "{r}");
    assert!((f(&face["source"]["interocular"]) - 40.0).abs() < 0.01);
    assert_eq!(f(&face["source"]["eyeLineDeg"]), 0.0);
    // Motion at 100 %, centred: sequence = source
    assert_eq!(face["sequence"]["eyeMid"], face["source"]["eyeMid"]);

    // scaled 200 % about the centre: sequence = 2 × (source − centre) + centre
    s.execute("effects.setParam", json!({"clip": c[0], "effect": "motion", "param": "scale", "value": 200.0})).unwrap();
    let r = s.execute("video.faces", json!({"clip": c[0]})).unwrap();
    let m = &r["faces"][0]["sequence"]["eyeMid"];
    assert!((f(&m["x"]) - (2.0 * (320.5 - 320.0) + 320.0)).abs() < 0.01 && (f(&m["y"]) - (2.0 * (150.5 - 180.0) + 180.0)).abs() < 0.01, "{r}");
    assert!((f(&r["faces"][0]["sequence"]["interocular"]) - 80.0).abs() < 0.01);

    // the program frame (rendered): same place, and mapped back to source
    let r = s.execute("video.faces", json!({"source": "program", "seconds": 0.0})).unwrap();
    let face = &r["faces"][0];
    assert!((f(&face["sequence"]["eyeMid"]["y"]) - 121.0).abs() < 1.5, "{r}");
    assert!((f(&face["source"]["eyeMid"]["y"]) - 150.5).abs() < 1.0, "{r}");

    // a region without the eyes: no face, not an error
    let r = s.execute("video.faces", json!({"clip": c[0], "region": {"x": 0, "y": 300, "w": 100, "h": 50}})).unwrap();
    assert_eq!(r["faces"], json!([]));
}

#[test]
fn align_eyes_match_clamps_then_auto_scales() {
    // incoming eyes 20 px lower: moving the clip up 20 px would show its bottom edge
    let (mut s, c) = setup(&[(300, 150), (310, 170)]);
    let audit = s.execute("video.auditJumpCuts", json!({})).unwrap();
    assert_eq!(audit["count"], 1);
    assert!((f(&audit["cuts"][0]["deltaY"]) - 20.0).abs() < 0.01, "{audit}");
    assert_eq!(audit["cuts"][0]["status"], "misaligned");

    let r = s.execute("video.alignEyes", json!({"from": c[0], "to": c[1]})).unwrap();
    let a = &r["applied"];
    assert_eq!(a["clamped"], true, "{r}");
    assert!((f(&a["residual"]["y"]) - 20.0).abs() < 0.01, "no move possible at 100 %: {r}");
    // smallest scale: max(T/e, (H−T)/(h−e)) = (360 − 150.5)/(360 − 170.5) = 110.55 % → 110.6
    assert!((f(&a["minScaleToAlign"]) - 110.6).abs() < 0.051, "{r}");

    let r = s.execute("video.alignEyes", json!({"from": c[0], "to": c[1], "autoScale": true})).unwrap();
    let a = &r["applied"];
    assert_eq!(a["clamped"], false, "{r}");
    assert!((f(&a["scale"]) - 110.6).abs() < 0.051 && f(&a["residual"]["y"]).abs() < 0.01, "{r}");
    // verified on the rendered program frames
    assert!(f(&r["verify"]["residual"]["y"]).abs() < 1.0, "{r}");
    let audit = s.execute("video.auditJumpCuts", json!({})).unwrap();
    assert_eq!(audit["misaligned"], 0, "{audit}");
}

#[test]
fn punch_in_lands_the_eyes_and_is_one_undo_step() {
    let (mut s, c) = setup(&[(300, 150), (300, 150)]);
    let before = s.active_sequence().unwrap().clone();
    let r = s.execute("video.alignEyes", json!({"from": c[0], "to": c[1], "mode": "punchIn", "punchPercent": 20})).unwrap();
    let a = &r["applied"];
    // Pat's formula: PositionY = eyeOut − s·(eyeIn − anchorY) = 150.5 − 1.2·(150.5 − 180) = 185.9
    assert!((f(&a["scale"]) - 120.0).abs() < 1e-6, "{r}");
    assert!((f(&a["position"][1]) - 185.9).abs() < 0.01, "{r}");
    assert!((f(&a["position"][0]) - 320.0).abs() < 1e-6, "X is kept: {r}");
    assert_eq!(a["clamped"], false);
    assert!(f(&r["verify"]["residual"]["y"]).abs() < 1.0, "{r}");
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(*s.active_sequence().unwrap(), before, "one undo step restores the clip");

    // axes xy: the incoming clip also moves sideways
    let (mut s, c) = setup(&[(300, 150), (320, 150)]);
    let r = s.execute("video.alignEyes", json!({"from": c[0], "to": c[1], "mode": "punchIn", "axes": "xy", "verify": false})).unwrap();
    assert!(f(&r["applied"]["residual"]["x"]).abs() < 0.01 && f(&r["applied"]["residual"]["y"]).abs() < 0.01, "{r}");
    assert!(r.get("verify").is_none());
}

#[test]
fn align_jump_cuts_chains_and_alternates() {
    let (mut s, c) = setup(&[(300, 150), (300, 160), (300, 150), (300, 140)]);
    let r = s.execute("video.alignJumpCuts", json!({"mode": "alternate"})).unwrap();
    assert_eq!(r["count"], 3, "{r}");
    let scales: Vec<f64> = r["cuts"].as_array().unwrap().iter().map(|x| f(&x["scale"])).collect();
    assert!((scales[0] - 110.0).abs() < 1e-6 && (scales[1] - 100.0).abs() < 1e-6 && (scales[2] - 110.0).abs() < 1e-6, "{r}");
    let audit = s.execute("video.auditJumpCuts", json!({"tolerance": 0.5})).unwrap();
    for cut in audit["cuts"].as_array().unwrap() {
        assert!(f(&cut["deltaY"]).abs() < 0.5, "{audit}");
    }
    // zoom reports the punch: in, out, in
    let z: Vec<f64> = audit["cuts"].as_array().unwrap().iter().map(|x| f(&x["zoom"])).collect();
    assert!((z[0] - 1.1).abs() < 0.01 && (z[1] - 1.0 / 1.1).abs() < 0.01, "{audit}");
    // one undo step for the whole pass
    s.execute("edit.undo", json!({})).unwrap();
    let audit = s.execute("video.auditJumpCuts", json!({})).unwrap();
    assert_eq!(audit["misaligned"], 3, "{audit}");
    let _ = c;

    // "match" leaves aligned cuts alone
    let (mut s, _) = setup(&[(300, 150), (300, 150)]);
    let r = s.execute("video.alignJumpCuts", json!({})).unwrap();
    assert_eq!((r["changed"].as_u64(), r["cuts"][0]["status"].as_str()), (Some(0), Some("alreadyAligned")), "{r}");
}

#[test]
fn hostile_params_are_errors_not_panics() {
    let (mut s, c) = setup(&[(300, 150), (300, 150)]);
    let audio = 999_998u64;
    for p in [
        json!({}),
        json!({"from": c[0]}),
        json!({"from": c[0], "to": c[0]}),
        json!({"from": c[0], "to": audio}),
        json!({"from": c[0], "to": 999_999}),
        json!({"from": c[0], "to": c[1], "mode": "zoom"}),
        json!({"from": c[0], "to": c[1], "scale": -5}),
        json!({"from": c[0], "to": c[1], "scale": 1e300}),
        json!({"from": "x", "to": [1]}),
    ] {
        assert!(s.execute("video.alignEyes", p.clone()).is_err(), "{p}");
    }
    for p in [
        json!({"clip": audio}),
        json!({"clip": c[0], "seconds": 1e9}),
        json!({"source": "elsewhere"}),
        json!({"clip": c[0], "region": {"x": 1e300, "y": -1e300, "w": 1, "h": 1}}),
    ] {
        let _ = s.execute("video.faces", p);
    }
    for p in [json!({"track": "V9"}), json!({"track": "A1"}), json!({"mode": "sideways"}), json!({"from": f64::MAX, "to": -1.0})] {
        let _ = s.execute("video.alignJumpCuts", p.clone());
        let _ = s.execute("video.auditJumpCuts", p);
    }
    let _ = s.execute("video.faces", json!({"maxFaces": -1, "minScore": "x", "scale": 0}));
}

#[test]
fn no_face_is_reported() {
    let dir = tmp_dir("face-none");
    let plain = dir.join("plain.png");
    image::RgbImage::from_pixel(W, H, image::Rgb([90, 90, 90])).save(&plain).unwrap();
    let a = still(&dir, "a.png", 300, 150);
    let (mut s, c) = sequence_of(&[a, plain]);
    let e = s.execute("video.alignEyes", json!({"from": c[0], "to": c[1]})).unwrap_err().to_string();
    assert!(e.contains("no face in the incoming clip"), "{e}");
    let audit = s.execute("video.auditJumpCuts", json!({})).unwrap();
    assert_eq!(audit["cuts"][0]["status"], "noFace");
    let r = s.execute("video.faces", json!({"clip": c[1]})).unwrap();
    assert_eq!(r["faces"], json!([]));
}

#[test]
fn face_model_is_described() {
    let mut s = Session::default();
    let r = s.execute("video.faceModel", json!({})).unwrap();
    assert_eq!(r["license"], "Apache-2.0");
    assert_eq!(r["size"], 229_032 + 1_242_398);
    for f in crate::face::FACE_MODEL.files {
        assert!(f.url.starts_with("https://storage.googleapis.com/mediapipe-assets/") && f.url.contains("?generation="), "{}", f.url);
        assert_eq!(f.sha256.len(), 64);
    }
}
