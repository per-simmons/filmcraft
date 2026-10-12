//! Face and eye measurement, and eye-line matching across jump cuts (`video.faces`,
//! `video.alignEyes`, `video.auditJumpCuts`, `video.alignJumpCuts`, `video.faceModel`,
//! `video.downloadFaceModel`).
//!
//! The editing technique (a talking-head pause removed, the incoming shot punched in about 10 %,
//! the eyes at the same screen height on both sides of the hard cut) needs the eye positions of the
//! last outgoing and first incoming frame. They are measured on the **source** frame of each clip
//! (decoded at full size) and carried to sequence pixels through the clip's Motion transform
//! ([`filmcraft_render::motion_matrix`]), so the incoming clip's new Position can be solved exactly:
//! for a source eye point `e`, `seq(e) = M(e)`, and the Position change is the difference between
//! where the eyes are and where they must land. `verify` then renders both program frames (every
//! track, adjustment layer and effect) and measures the eyes there: that residual is the honest one.
//!
//! The detector is [`Session::face_detector`] when a host or test installed one, else MediaPipe
//! BlazeFace + Face Mesh ([`filmcraft_vision`]) from `<data dir>/models/mediapipe-face`, downloaded
//! on request (`video.downloadFaceModel`, or `download: true`; feature `speech-download`).

use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

use filmcraft_geom::{Affine, Vec2};
use filmcraft_project::{ClipId, ParamValue, Sequence, TrackItem, TrackKind};
use filmcraft_speech::models::{ModelFile, ModelInfo};
use filmcraft_time::Tick;
use filmcraft_vision::{BBox, DetectOptions, DetectorKind, Face, FaceModels, Image, Point};

use crate::commands::{CommandSpec, bad, bool_p, f64_p, has_seq, str_p, time_p, track_p, u64_p};
use crate::{EngineError, Result, Session};

/// A face detector the engine can use (installed in [`Session::face_detector`] by hosts and tests).
pub trait FaceDetector: Send + Sync {
    /// Faces in `img`, the best (largest, mesh-confirmed) first; picture pixels.
    fn detect(&self, img: &Image, opts: &DetectOptions) -> std::result::Result<Vec<Face>, String>;
    /// A short name for reports.
    fn id(&self) -> String;
}

impl FaceDetector for FaceModels {
    fn detect(&self, img: &Image, opts: &DetectOptions) -> std::result::Result<Vec<Face>, String> {
        FaceModels::detect(self, img, opts).map_err(|e| e.to_string())
    }
    fn id(&self) -> String {
        FACE_MODEL.id.into()
    }
}

const DETECTOR_FILE: &str = "face_detection_short_range.tflite";
const MESH_FILE: &str = "face_landmark.tflite";

/// MediaPipe's face detector and Face Mesh, pinned by GCS object generation and SHA-256.
///
/// Licence: Apache-2.0 (the MediaPipe repository and the model cards "BlazeFace (short range)" and
/// "MediaPipe Face Mesh"; <https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE>).
pub static FACE_MODEL: ModelInfo = ModelInfo {
    id: "mediapipe-face",
    name: "MediaPipe face detector + Face Mesh",
    multilingual: false,
    description: "BlazeFace short-range face detector (128 px) and the 468-point Face Mesh. 1.5 MB. Finds faces and eye centres for eye-line matching across jump cuts.",
    license: "Apache-2.0",
    license_url: "https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE",
    author: "Google (MediaPipe)",
    source: "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker",
    files: &[
        ModelFile {
            name: DETECTOR_FILE,
            url: "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite?generation=1677522680211413",
            sha256: "3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633",
            size: 229_032,
        },
        ModelFile {
            name: MESH_FILE,
            url: "https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite?generation=1676323214951585",
            sha256: "1055cb9d4a9ca8b8c688902a3a5194311138ba256bcc94e336d8373a5f30c814",
            size: 1_242_398,
        },
    ],
};

/// The loaded MediaPipe models, shared by every session (loading parses 1.5 MB).
static LOADED: Mutex<Option<Arc<FaceModels>>> = Mutex::new(None);

fn models_installed() -> bool {
    crate::transcript::models_dir().is_some_and(|d| filmcraft_speech::models::installed(&d, &FACE_MODEL))
}

fn can_download(_: &Session) -> std::result::Result<(), String> {
    if cfg!(feature = "speech-download") { Ok(()) } else { Err(NO_DOWNLOAD.into()) }
}

const NO_DOWNLOAD: &str = "model downloads are not available in this build (built without the `speech-download` feature)";

fn download_models() -> Result<()> {
    let dir = crate::transcript::models_dir().ok_or_else(|| EngineError::Other("no data directory for models".into()))?;
    #[cfg(feature = "speech-download")]
    {
        filmcraft_speech::models::download(&dir, &FACE_MODEL, &mut |_, _, _| true).map_err(|e| EngineError::Other(format!("face model: {e}")))
    }
    #[cfg(not(feature = "speech-download"))]
    {
        let _ = dir;
        Err(EngineError::Other(NO_DOWNLOAD.into()))
    }
}

/// The detector to use: the installed one, else the MediaPipe models (downloaded first when
/// `download: true`).
fn detector(s: &Session, p: &Value, cmd: &str) -> Result<Arc<dyn FaceDetector>> {
    if let Some(d) = &s.face_detector {
        return Ok(d.clone());
    }
    let mut slot = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(m) = slot.as_ref() {
        return Ok(m.clone());
    }
    if !models_installed() {
        if bool_p(p, "download").unwrap_or(false) {
            download_models()?;
        } else {
            return Err(bad(
                cmd,
                format!(
                    "the face model is not downloaded: run video.downloadFaceModel (or pass download:true); {} ({:.1} MB, {}, {})",
                    FACE_MODEL.name,
                    FACE_MODEL.size() as f64 / 1e6,
                    FACE_MODEL.license,
                    FACE_MODEL.source
                ),
            ));
        }
    }
    let dir = crate::transcript::models_dir().ok_or_else(|| EngineError::Other("no data directory for models".into()))?;
    let dir = filmcraft_speech::models::model_dir(&dir, &FACE_MODEL);
    let read = |f: &str| std::fs::read(dir.join(f)).map_err(|e| EngineError::Other(format!("{cmd}: can't read {f}: {e}")));
    let m = FaceModels::load(&read(DETECTOR_FILE)?, DetectorKind::ShortRange, Some(&read(MESH_FILE)?))
        .map_err(|e| EngineError::Other(format!("{cmd}: face model: {e}")))?;
    let m = Arc::new(m);
    *slot = Some(m.clone());
    Ok(m)
}

// ---------------------------------------------------------------------------------------------
// Measuring
// ---------------------------------------------------------------------------------------------

fn r2(v: f64) -> f64 {
    if v.is_finite() { (v * 100.0).round() / 100.0 } else { 0.0 }
}

fn pt(p: Point) -> Value {
    json!({"x": r2(p.x), "y": r2(p.y)})
}

fn secs(t: Tick) -> f64 {
    (t.seconds() * 1000.0).round() / 1000.0
}

fn apply(m: &Affine, p: Point) -> Point {
    let v = m.apply(Vec2::new(p.x, p.y));
    Point::new(v.x, v.y)
}

/// A face's geometry in one coordinate space.
fn geometry(f: &Face, m: &Affine) -> Value {
    let (l, r) = (apply(m, f.left_eye), apply(m, f.right_eye));
    let b = m.bounds(&filmcraft_geom::Rect::new(f.bbox.x, f.bbox.y, f.bbox.w, f.bbox.h));
    let mid = Point::new((l.x + r.x) / 2.0, (l.y + r.y) / 2.0);
    json!({
        "box": {"x": r2(b.x), "y": r2(b.y), "w": r2(b.w), "h": r2(b.h)},
        "leftEye": pt(l), "rightEye": pt(r), "eyeMid": pt(mid),
        "eyeLineDeg": r2((r.y - l.y).atan2(r.x - l.x).to_degrees()),
        "interocular": r2((r.x - l.x).hypot(r.y - l.y)),
    })
}

fn region_p(p: &Value) -> Option<BBox> {
    let g = p.get("region")?;
    let v = |k: &str| g.get(k).and_then(Value::as_f64).filter(|v| v.is_finite());
    Some(BBox { x: v("x")?, y: v("y")?, w: v("w")?, h: v("h")? })
}

fn detect_options(p: &Value) -> DetectOptions {
    let d = DetectOptions::default();
    DetectOptions {
        min_score: f64_p(p, "minScore").filter(|v| v.is_finite()).map(|v| v.clamp(0.05, 0.99) as f32).unwrap_or(d.min_score),
        max_faces: u64_p(p, "maxFaces").map(|v| v.clamp(1, 16) as usize).unwrap_or(d.max_faces),
        region: region_p(p),
        tiles: bool_p(p, "tiles").unwrap_or(true),
    }
}

/// One clip measured at a sequence time: faces in source pixels and the Motion transform that
/// carries them to sequence pixels.
struct ClipFaces {
    clip: TrackItem,
    t: Tick,
    source_time: Tick,
    source_size: (u32, u32),
    motion: Affine,
    faces: Vec<Face>,
}

/// Decode `clip`'s source frame shown at sequence time `t` and find its faces.
fn measure_clip(
    s: &Session,
    q: &Sequence,
    clip: &TrackItem,
    t: Tick,
    det: &dyn FaceDetector,
    opts: &DetectOptions,
    scale: f32,
    cmd: &str,
) -> Result<ClipFaces> {
    let src = s.source(clip.item).ok_or_else(|| bad(cmd, format!("clip {} has no picture (offline or not media)", clip.id.0)))?;
    let size = filmcraft_render::source_size(&s.project, clip.item).ok_or_else(|| bad(cmd, format!("clip {} has no picture", clip.id.0)))?;
    let source_time = clip.source_time_at(t);
    let f = src
        .video_frame(filmcraft_media::FrameRequest { time: source_time, scale })
        .map_err(|e| EngineError::Other(format!("{cmd}: can't decode clip {} at {:.3} s: {e}", clip.id.0, t.seconds())))?;
    let (w, h) = (f.width as usize, f.height as usize);
    let rgba = f.to_rgba8();
    let img = Image::new(w, h, &rgba).map_err(|e| EngineError::Other(format!("{cmd}: {e}")))?;
    // decoded pixels → source pixels (the frame may come back reduced)
    let (fx, fy) = (f64::from(size.0) / w.max(1) as f64, f64::from(size.1) / h.max(1) as f64);
    let mut o = *opts;
    o.region = opts.region.map(|r| BBox { x: r.x / fx, y: r.y / fy, w: r.w / fx, h: r.h / fy });
    let mut faces = det.detect(&img, &o).map_err(|e| EngineError::Other(format!("{cmd}: {e}")))?;
    let to_src = Affine::scale(fx, fy);
    for face in &mut faces {
        face.left_eye = apply(&to_src, face.left_eye);
        face.right_eye = apply(&to_src, face.right_eye);
        face.bbox = BBox { x: face.bbox.x * fx, y: face.bbox.y * fy, w: face.bbox.w * fx, h: face.bbox.h * fy };
        face.mesh.clear();
    }
    let motion = filmcraft_render::motion_matrix(q, clip, size, clip.effect_time_at(t));
    Ok(ClipFaces { clip: clip.clone(), t, source_time, source_size: size, motion, faces })
}

/// The program frame at `t`, faces in sequence pixels.
fn measure_program(s: &Session, t: Tick, det: &dyn FaceDetector, opts: &DetectOptions, cmd: &str) -> Result<Vec<Face>> {
    let img = s.try_render_program_at(1.0, t)?;
    let rgba = img.over_black_rgba8();
    let pic = Image::new(img.w, img.h, &rgba).map_err(|e| EngineError::Other(format!("{cmd}: {e}")))?;
    det.detect(&pic, opts).map_err(|e| EngineError::Other(format!("{cmd}: {e}")))
}

fn face_json(f: &Face, source: Option<&Affine>, seq: &Affine) -> Value {
    json!({
        "score": r2(f64::from(f.score)),
        "meshScore": f.mesh_score.map(|v| r2(f64::from(v))),
        "method": f.method,
        "sequence": geometry(f, seq),
        "source": source.map(|m| geometry(f, m)),
    })
}

fn video_clip(q: &Sequence, id: ClipId, cmd: &str, what: &str) -> Result<TrackItem> {
    q.find_item(id)
        .filter(|(t, _)| q.track(*t).is_some_and(|t| t.kind == TrackKind::Video))
        .map(|(_, i)| i.clone())
        .ok_or_else(|| bad(cmd, format!("`{what}`: no video clip {} in the active sequence", id.0)))
}

/// The topmost enabled video clip at `t`.
fn top_clip(q: &Sequence, t: Tick) -> Option<TrackItem> {
    q.video_tracks.iter().rev().flat_map(|tr| tr.items.iter()).find(|i| i.enabled && i.start <= t && t < i.end()).cloned()
}

fn frame_scale(p: &Value) -> f32 {
    f64_p(p, "scale").filter(|v| v.is_finite() && *v > 0.0).map(|v| v.min(1.0) as f32).unwrap_or(1.0)
}

fn faces_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "video.faces";
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?.clone();
    let clip = match u64_p(p, "clip").map(ClipId) {
        Some(c) => Some(video_clip(&q, c, cmd, "clip")?),
        None => None,
    };
    let rate = s.sequence_rate();
    let t = time_p(s, p, "").unwrap_or_else(|| match &clip {
        Some(c) if !(c.start <= s.playhead() && s.playhead() < c.end()) => c.start,
        _ => s.playhead(),
    });
    let t = rate.snap(t.max(Tick::ZERO));
    let source = str_p(p, "source").unwrap_or(if clip.is_some() { "clip" } else { "program" });
    let det = detector(s, p, cmd)?;
    let opts = detect_options(p);
    match source {
        "clip" => {
            let c = match clip.or_else(|| top_clip(&q, t)) {
                Some(c) => c,
                None => return Err(bad(cmd, format!("no video clip at {:.3} s", t.seconds()))),
            };
            if !(c.start <= t && t < c.end()) {
                return Err(bad(cmd, format!("{:.3} s is outside clip {} ({:.3}–{:.3} s)", t.seconds(), c.id.0, c.start.seconds(), c.end().seconds())));
            }
            let m = measure_clip(s, &q, &c, t, det.as_ref(), &opts, frame_scale(p), cmd)?;
            Ok(json!({
                "source": "clip", "clip": c.id.0, "seconds": secs(t), "frame": rate.frame_at(t),
                "sourceSeconds": secs(m.source_time), "sourceSize": [m.source_size.0, m.source_size.1],
                "sequenceSize": [q.settings.width, q.settings.height], "detector": det.id(),
                "faces": m.faces.iter().map(|f| face_json(f, Some(&Affine::IDENTITY), &m.motion)).collect::<Vec<_>>(),
                "note": "leftEye / rightEye = the eye on the left / right of the picture (the subject's right / left); eye centres are eye-corner midpoints; `sequence` = through the clip's Motion (not other tracks or effects; use source:\"program\" for the composite)",
            }))
        }
        "program" => {
            let faces = measure_program(s, t, det.as_ref(), &opts, cmd)?;
            // source pixels of the topmost clip, when its Motion can be inverted
            let top = top_clip(&q, t);
            let inv = top.as_ref().and_then(|c| {
                let size = filmcraft_render::source_size(&s.project, c.item)?;
                filmcraft_render::motion_matrix(&q, c, size, c.effect_time_at(t)).inverse()
            });
            Ok(json!({
                "source": "program", "clip": top.as_ref().map(|c| c.id.0), "seconds": secs(t), "frame": rate.frame_at(t),
                "sequenceSize": [q.settings.width, q.settings.height], "detector": det.id(),
                "faces": faces.iter().map(|f| face_json(f, inv.as_ref(), &Affine::IDENTITY)).collect::<Vec<_>>(),
                "note": "measured on the rendered program frame (every track and effect); `source` maps back through the topmost clip's Motion",
            }))
        }
        o => Err(bad(cmd, format!("source must be \"clip\" or \"program\", got {o:?}"))),
    }
}

// ---------------------------------------------------------------------------------------------
// Aligning
// ---------------------------------------------------------------------------------------------

/// Which axes to match.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Axes {
    Y,
    XY,
}

/// How the incoming clip's scale is chosen.
#[derive(Clone, Copy, PartialEq, Debug)]
enum ScaleMode {
    /// Keep its scale (or the explicit `scale`).
    Keep,
    /// The outgoing clip's scale × (1 + punch).
    PunchIn(f64),
    /// Scale so the faces are the same size (interocular distance).
    MatchSize,
    /// Explicit Motion Scale percent.
    Percent(f64),
}

#[derive(Clone, Copy)]
struct AlignOpts {
    axes: Axes,
    clamp: bool,
    auto_scale: bool,
}

/// The computed change for one incoming clip.
#[derive(Clone, Debug)]
struct Plan {
    position: Vec2,
    /// Motion Scale (percent) and Scale Width (when not uniform).
    scale: f64,
    scale_width: Option<f64>,
    target: Point,
    predicted: Point,
    clamped: bool,
    min_scale: Option<f64>,
    warnings: Vec<String>,
}

fn motion_values(q: &Sequence, it: &TrackItem, mt: Tick) -> (Vec2, f64, Option<f64>, f64) {
    // the defaults `filmcraft_render::motion_matrix` uses for a parameter that isn't set
    let centre = Vec2::new(f64::from(q.settings.width) / 2.0, f64::from(q.settings.height) / 2.0);
    match it.effect("motion").filter(|m| m.enabled) {
        Some(m) => {
            let uniform = m.param("uniform_scale").and_then(|p| p.value.as_bool()).unwrap_or(true);
            let p = m.vec2_at("position", mt);
            let pos = if m.param("position").is_some() && p.x.is_finite() && p.y.is_finite() { p } else { centre };
            (pos, m.f64_at("scale", mt), (!uniform).then(|| m.f64_at("scale_width", mt)), m.f64_at("rotation", mt))
        }
        None => (centre, 100.0, None, 0.0),
    }
}

/// Set a Motion parameter to a static value (keyframes removed).
fn set_static(it: &mut TrackItem, k: &str, v: ParamValue) -> bool {
    let Some(m) = it.effect_mut("motion") else { return false };
    m.enabled = true;
    match m.param_mut(k) {
        Some(prm) => {
            let had = !prm.keyframes.is_empty();
            prm.keyframes.clear();
            prm.value = v;
            had
        }
        None => {
            m.params.insert(k.to_string(), filmcraft_project::Param::new(v));
            false
        }
    }
}

/// A copy of `it` with the Motion values of `plan` (for computing transforms).
fn with_plan(it: &TrackItem, position: Vec2, scale: f64, scale_width: Option<f64>) -> TrackItem {
    let mut c = it.clone();
    set_static(&mut c, "position", ParamValue::Vec2(position));
    set_static(&mut c, "scale", ParamValue::Float(scale));
    if let Some(w) = scale_width {
        set_static(&mut c, "scale_width", ParamValue::Float(w));
    }
    c
}

/// Effective vertical scale of a Motion matrix (source px → sequence px).
fn y_scale(m: &Affine) -> f64 {
    m.c.hypot(m.d)
}

/// Solve the incoming clip's Position (and Scale) so its eye midpoint `eye_b` (source px) lands on
/// `target` (sequence px) at effect time `mt`.
#[allow(clippy::too_many_arguments)]
fn plan(
    q: &Sequence,
    b: &TrackItem,
    size: (u32, u32),
    mt: Tick,
    eye_b: Point,
    iod_b: f64,
    target: Point,
    a_eff_scale: f64,
    a_iod_seq: f64,
    mode: ScaleMode,
    o: AlignOpts,
) -> Option<Plan> {
    let (pos0, scale0, sw0, rot) = motion_values(q, b, mt);
    let m0 = filmcraft_render::motion_matrix(q, b, size, mt);
    let eff0 = y_scale(&m0);
    if !(eff0 > 1e-9 && scale0.is_finite() && scale0 > 0.0) {
        return None;
    }
    // percent of Motion Scale per unit of effective scale (Scale to Frame Size, nesting)
    let pct_per_eff = scale0 / eff0;
    let mut warnings = Vec::new();
    let mut scale = match mode {
        ScaleMode::Keep => scale0,
        ScaleMode::Percent(p) => p,
        ScaleMode::PunchIn(k) => a_eff_scale * (1.0 + k) * pct_per_eff,
        ScaleMode::MatchSize if iod_b > 1e-6 => a_iod_seq / iod_b * pct_per_eff,
        ScaleMode::MatchSize => {
            warnings.push("no interocular distance in the incoming frame: scale kept".into());
            scale0
        }
    };
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    scale = scale.clamp(1.0, 10_000.0);
    let rotated = rot.abs() > 1e-6;
    let solve = |scale: f64| -> (Vec2, Option<f64>, Point, Option<(f64, f64)>, Option<(f64, f64)>) {
        let sw = sw0.map(|w| w * scale / scale0);
        let m = filmcraft_render::motion_matrix(q, &with_plan(b, pos0, scale, sw), size, mt);
        let e = m.apply(Vec2::new(eye_b.x, eye_b.y));
        let d = Vec2::new(if o.axes == Axes::XY { target.x - e.x } else { 0.0 }, target.y - e.y);
        // translations that keep the frame covered: [lo, hi] per axis
        let bounds = m.bounds(&filmcraft_geom::Rect::new(0.0, 0.0, f64::from(size.0), f64::from(size.1)));
        let (fw, fh) = (f64::from(q.settings.width), f64::from(q.settings.height));
        let ry = (fh - (bounds.y + bounds.h), -bounds.y);
        let rx = (fw - (bounds.x + bounds.w), -bounds.x);
        (Vec2::new(pos0.x + d.x, pos0.y + d.y), sw, Point::new(e.x + d.x, e.y + d.y), (ry.0 <= ry.1).then_some(ry), (rx.0 <= rx.1).then_some(rx))
    };
    let fits = |scale: f64| -> (bool, Vec2, Option<f64>, Point, bool) {
        let (mut pos, sw, mut pred, ry, rx) = solve(scale);
        let mut clamped = false;
        if o.clamp && !rotated {
            let dy = pos.y - pos0.y;
            if let Some((lo, hi)) = ry {
                let c = dy.clamp(lo, hi);
                if (c - dy).abs() > 1e-9 {
                    clamped = true;
                    pred.y += c - dy;
                    pos.y = pos0.y + c;
                }
            }
            let dx = pos.x - pos0.x;
            if o.axes == Axes::XY
                && let Some((lo, hi)) = rx
            {
                let c = dx.clamp(lo, hi);
                if (c - dx).abs() > 1e-9 {
                    clamped = true;
                    pred.x += c - dx;
                    pos.x = pos0.x + c;
                }
            }
        }
        (!clamped, pos, sw, pred, clamped)
    };
    let (ok, mut position, mut scale_width, mut predicted, mut clamped) = fits(scale);
    if o.clamp && rotated {
        warnings.push("the clip is rotated: borders are not checked".into());
    }
    // the smallest scale that aligns without exposing a border
    let mut min_scale = None;
    if !ok {
        let (mut lo, mut hi) = (scale, scale * 4.0);
        if fits(hi).0 {
            for _ in 0..40 {
                let mid = (lo + hi) / 2.0;
                if fits(mid).0 { hi = mid } else { lo = mid }
            }
            let up = (hi * 10.0).ceil() / 10.0;
            min_scale = Some(up);
            if o.auto_scale {
                scale = up;
                let r = fits(up);
                (position, scale_width, predicted, clamped) = (r.1, r.2, r.3, r.4);
            }
        }
    }
    if clamped {
        warnings.push(match min_scale {
            Some(m) if !o.auto_scale => format!("held back so no border shows; Scale {m:.1} % (or autoScale:true) aligns exactly"),
            _ => "held back so no border shows".into(),
        });
    }
    Some(Plan { position, scale, scale_width, target, predicted, clamped, min_scale, warnings })
}

/// Pick the face to align: the largest mesh face, or the one nearest `near` (sequence px).
fn pick(faces: &[Face], m: &Affine, near: Option<Point>) -> Option<Face> {
    match near {
        None => faces.first().cloned(),
        Some(n) => faces
            .iter()
            .min_by(|a, b| {
                let d = |f: &Face| {
                    let e = apply(m, f.eye_mid());
                    (e.x - n.x).hypot(e.y - n.y)
                };
                d(a).total_cmp(&d(b))
            })
            .cloned(),
    }
}

fn scale_mode(p: &Value, cmd: &str, default: &str) -> Result<ScaleMode> {
    let punch = f64_p(p, "punchPercent").filter(|v| v.is_finite()).unwrap_or(10.0).clamp(-50.0, 300.0) / 100.0;
    if let Some(v) = f64_p(p, "scale") {
        if !(v.is_finite() && v > 0.0 && v <= 10_000.0) {
            return Err(bad(cmd, "scale is a Motion Scale percent (> 0)"));
        }
        return Ok(ScaleMode::Percent(v));
    }
    Ok(match str_p(p, "mode").unwrap_or(default) {
        "match" => ScaleMode::Keep,
        "punchIn" => ScaleMode::PunchIn(punch),
        "matchSize" => ScaleMode::MatchSize,
        o => return Err(bad(cmd, format!("mode must be \"match\", \"punchIn\" or \"matchSize\", got {o:?}"))),
    })
}

fn align_opts(p: &Value) -> AlignOpts {
    AlignOpts {
        axes: if str_p(p, "axes") == Some("xy") { Axes::XY } else { Axes::Y },
        clamp: bool_p(p, "clampBorders").unwrap_or(true),
        auto_scale: bool_p(p, "autoScale").unwrap_or(false),
    }
}

/// The measurements of one cut: the outgoing clip's last frame, the incoming clip's first.
struct Cut {
    a: ClipFaces,
    b: ClipFaces,
}

fn measure_cut(s: &Session, q: &Sequence, a: &TrackItem, b: &TrackItem, det: &dyn FaceDetector, opts: &DetectOptions, scale: f32, cmd: &str) -> Result<Cut> {
    let fd = s.sequence_rate().frame_duration().max(Tick(1));
    let ta = (a.end() - fd).max(a.start);
    Ok(Cut { a: measure_clip(s, q, a, ta, det, opts, scale, cmd)?, b: measure_clip(s, q, b, b.start, det, opts, scale, cmd)? })
}

/// Eye midpoints (sequence px) of a cut through the current transforms: (A, B, A face, B face).
fn cut_eyes(c: &Cut) -> Option<(Point, Point, Face, Face)> {
    let fa = pick(&c.a.faces, &c.a.motion, None)?;
    let ea = apply(&c.a.motion, fa.eye_mid());
    let fb = pick(&c.b.faces, &c.b.motion, Some(ea))?;
    Some((ea, apply(&c.b.motion, fb.eye_mid()), fa, fb))
}

fn plan_json(p: &Plan, b: &TrackItem) -> Value {
    json!({
        "clip": b.id.0,
        "position": [r2(p.position.x), r2(p.position.y)],
        "scale": r2(p.scale),
        "scaleWidth": p.scale_width.map(r2),
        "target": pt(p.target),
        "predicted": pt(p.predicted),
        "residual": {"x": r2(p.predicted.x - p.target.x), "y": r2(p.predicted.y - p.target.y)},
        "clamped": p.clamped,
        "minScaleToAlign": p.min_scale.map(r2),
        "warnings": p.warnings,
    })
}

/// Write the plans (one undo step).
fn apply_plans(s: &mut Session, label: &str, plans: &[(ClipId, Plan)]) -> Result<Vec<u64>> {
    s.edit_sequence(label, |q, _, _| {
        let mut keys_removed = Vec::new();
        for (id, p) in plans {
            let Some((_, it)) = q.find_item_mut(*id) else { continue };
            let mut had = set_static(it, "position", ParamValue::Vec2(p.position));
            had |= set_static(it, "scale", ParamValue::Float(p.scale));
            if let Some(w) = p.scale_width {
                had |= set_static(it, "scale_width", ParamValue::Float(w));
            }
            if had {
                keys_removed.push(id.0);
            }
        }
        Ok(keys_removed)
    })
}

/// Measure the rendered program at the cut: (A eye mid, B eye mid) in sequence px.
fn verify_cut(s: &Session, a_end: Tick, b_start: Tick, near: Point, det: &dyn FaceDetector, opts: &DetectOptions, cmd: &str) -> Result<Value> {
    let fd = s.sequence_rate().frame_duration().max(Tick(1));
    let fa = measure_program(s, a_end - fd, det, opts, cmd)?;
    let fb = measure_program(s, b_start, det, opts, cmd)?;
    let ea = pick(&fa, &Affine::IDENTITY, Some(near)).map(|f| f.eye_mid());
    let eb = pick(&fb, &Affine::IDENTITY, Some(near)).map(|f| f.eye_mid());
    Ok(match (ea, eb) {
        (Some(a), Some(b)) => json!({"outgoing": pt(a), "incoming": pt(b), "residual": {"x": r2(b.x - a.x), "y": r2(b.y - a.y)}}),
        _ => json!({"error": "no face found on a rendered program frame", "outgoingFound": ea.is_some(), "incomingFound": eb.is_some()}),
    })
}

fn align_eyes(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "video.alignEyes";
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?.clone();
    let a = video_clip(&q, u64_p(p, "from").map(ClipId).ok_or_else(|| bad(cmd, "`from` (the outgoing clip) is required"))?, cmd, "from")?;
    let b = video_clip(&q, u64_p(p, "to").map(ClipId).ok_or_else(|| bad(cmd, "`to` (the incoming clip) is required"))?, cmd, "to")?;
    if a.id == b.id {
        return Err(bad(cmd, "`from` and `to` must be different clips"));
    }
    let mode = scale_mode(p, cmd, "match")?;
    let o = align_opts(p);
    let det = detector(s, p, cmd)?;
    let opts = detect_options(p);
    let cut = measure_cut(s, &q, &a, &b, det.as_ref(), &opts, frame_scale(p), cmd)?;
    let Some((ea, eb_before, fa, fb)) = cut_eyes(&cut) else {
        let which = if cut.a.faces.is_empty() { "outgoing" } else { "incoming" };
        let c = if cut.a.faces.is_empty() { &cut.a } else { &cut.b };
        return Err(bad(
            cmd,
            format!(
                "no face in the {which} clip {} at {:.3} s (source {:.3} s); pass `region` (source px) or check the frame",
                c.clip.id.0,
                c.t.seconds(),
                c.source_time.seconds()
            ),
        ));
    };
    let a_iod = {
        let (l, r) = (apply(&cut.a.motion, fa.left_eye), apply(&cut.a.motion, fa.right_eye));
        (r.x - l.x).hypot(r.y - l.y)
    };
    let mt = b.effect_time_at(cut.b.t);
    let pl = plan(&q, &b, cut.b.source_size, mt, fb.eye_mid(), fb.interocular(), ea, y_scale(&cut.a.motion), a_iod, mode, o)
        .ok_or_else(|| bad(cmd, "the incoming clip's Motion can't be solved (zero scale?)"))?;
    let keys = apply_plans(s, "Align Eyes", &[(b.id, pl.clone())])?;
    let mut out = json!({
        "from": a.id.0, "to": b.id.0,
        "cutSeconds": secs(b.start), "adjacent": a.end() == b.start,
        "outgoing": {"seconds": secs(cut.a.t), "sourceSeconds": secs(cut.a.source_time), "eyeMid": pt(ea), "face": face_json(&fa, Some(&Affine::IDENTITY), &cut.a.motion)},
        "incoming": {"seconds": secs(cut.b.t), "sourceSeconds": secs(cut.b.source_time), "eyeMidBefore": pt(eb_before), "face": face_json(&fb, Some(&Affine::IDENTITY), &cut.b.motion)},
        "applied": plan_json(&pl, &b),
        "keyframesRemoved": !keys.is_empty(),
        "detector": det.id(),
    });
    if bool_p(p, "verify").unwrap_or(true) {
        out["verify"] = verify_cut(s, a.end(), b.start, ea, det.as_ref(), &opts, cmd)?;
    }
    Ok(out)
}

/// Adjacent video clip pairs (A ends where B starts) on `track` within `[from, to]`.
fn cuts_on(q: &Sequence, track: usize, from: Tick, to: Tick, same_media: bool) -> Vec<(TrackItem, TrackItem)> {
    let Some(tr) = q.video_tracks.get(track) else { return Vec::new() };
    let mut items: Vec<&TrackItem> = tr.items.iter().filter(|i| i.enabled).collect();
    items.sort_by_key(|i| i.start);
    items
        .windows(2)
        .filter_map(|w| match w {
            [a, b] if a.end() == b.start && b.start >= from && b.start <= to && (!same_media || a.item == b.item) => Some(((*a).clone(), (*b).clone())),
            _ => None,
        })
        .collect()
}

const MAX_CUTS: usize = 200;

fn track_range(s: &Session, p: &Value, q: &Sequence, cmd: &str) -> Result<(usize, Tick, Tick)> {
    let tid = track_p(s, p, "track", cmd)?;
    let track = match tid {
        Some(t) => q.video_tracks.iter().position(|x| x.id == t).ok_or_else(|| bad(cmd, "`track` must be a video track"))?,
        None => 0,
    };
    let from = time_p(s, p, "from").or_else(|| f64_p(p, "from").map(Tick::from_seconds_f64)).unwrap_or(Tick::ZERO);
    let to = time_p(s, p, "to").or_else(|| f64_p(p, "to").map(Tick::from_seconds_f64)).unwrap_or(q.duration());
    Ok((track, from, to))
}

fn audit_jump_cuts(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "video.auditJumpCuts";
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?.clone();
    let (track, from, to) = track_range(s, p, &q, cmd)?;
    let cuts = cuts_on(&q, track, from, to, bool_p(p, "sameMedia").unwrap_or(false));
    if cuts.len() > MAX_CUTS {
        return Err(bad(cmd, format!("{} cuts: at most {MAX_CUTS} per call (narrow from/to)", cuts.len())));
    }
    let tol = f64_p(p, "tolerance").filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(2.0);
    let det = detector(s, p, cmd)?;
    let opts = detect_options(p);
    let scale = frame_scale(p);
    let mut rows = Vec::new();
    let (mut misaligned, mut worst) = (0usize, 0f64);
    for (a, b) in &cuts {
        let c = measure_cut(s, &q, a, b, det.as_ref(), &opts, scale, cmd)?;
        let (sa, sb) = (y_scale(&c.a.motion), y_scale(&c.b.motion));
        let row = match cut_eyes(&c) {
            Some((ea, eb, fa, fb)) => {
                let dy = eb.y - ea.y;
                let ok = dy.abs() <= tol;
                misaligned += usize::from(!ok);
                worst = worst.max(dy.abs());
                let ia = fa.interocular() * sa;
                let ib = fb.interocular() * sb;
                json!({
                    "seconds": secs(b.start), "from": a.id.0, "to": b.id.0,
                    "outgoingEyeMid": pt(ea), "incomingEyeMid": pt(eb),
                    "deltaY": r2(dy), "deltaX": r2(eb.x - ea.x),
                    "zoom": if ia > 1e-6 { r2(ib / ia) } else { 0.0 },
                    "scaleRatio": if sa > 1e-9 { r2(sb / sa) } else { 0.0 },
                    "status": if ok { "aligned" } else { "misaligned" },
                })
            }
            None => json!({
                "seconds": secs(b.start), "from": a.id.0, "to": b.id.0, "status": "noFace",
                "outgoingFaces": c.a.faces.len(), "incomingFaces": c.b.faces.len(),
            }),
        };
        rows.push(row);
    }
    Ok(json!({
        "track": format!("V{}", track + 1), "cuts": rows, "count": cuts.len(), "misaligned": misaligned,
        "worstDeltaY": r2(worst), "tolerance": tol, "detector": det.id(),
        "note": "sequence px through each clip's Motion: deltaY = incoming − outgoing eye-midpoint height (positive: the eyes drop at the cut); zoom = incoming/outgoing face size on screen",
    }))
}

fn align_jump_cuts(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "video.alignJumpCuts";
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?.clone();
    let (track, from, to) = track_range(s, p, &q, cmd)?;
    let cuts = cuts_on(&q, track, from, to, bool_p(p, "sameMedia").unwrap_or(false));
    if cuts.len() > MAX_CUTS {
        return Err(bad(cmd, format!("{} cuts: at most {MAX_CUTS} per call (narrow from/to)", cuts.len())));
    }
    let tol = f64_p(p, "tolerance").filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(0.5);
    let o = align_opts(p);
    // "match": keep each scale; "alternate": punch in on every other cut (100 → 110 → 100 …)
    let alternate = match str_p(p, "mode").unwrap_or("match") {
        "match" => false,
        "alternate" => true,
        m => return Err(bad(cmd, format!("mode must be \"match\" or \"alternate\", got {m:?}"))),
    };
    let punch = f64_p(p, "punchPercent").filter(|v| v.is_finite()).unwrap_or(10.0).clamp(-50.0, 300.0) / 100.0;
    let det = detector(s, p, cmd)?;
    let opts = detect_options(p);
    let scale = frame_scale(p);
    // measure every cut first (source eyes don't depend on the transforms being changed)
    let mut measured = Vec::new();
    for (a, b) in &cuts {
        measured.push(measure_cut(s, &q, a, b, det.as_ref(), &opts, scale, cmd)?);
    }
    // the reference (unpunched) on-screen scale: the first outgoing clip's
    let base = measured.first().map(|c| y_scale(&c.a.motion)).unwrap_or(1.0);
    let mut work = q.clone();
    let mut plans: Vec<(ClipId, Plan)> = Vec::new();
    let mut rows = Vec::new();
    for c in &measured {
        let (a_id, b_id) = (c.a.clip.id, c.b.clip.id);
        let (Some((_, a_now)), Some((_, b_now))) = (work.find_item(a_id), work.find_item(b_id)) else { continue };
        let (a_now, b_now) = (a_now.clone(), b_now.clone());
        // A's transform as already planned in this pass
        let ma = filmcraft_render::motion_matrix(&work, &a_now, c.a.source_size, a_now.effect_time_at(c.a.t));
        let mb = filmcraft_render::motion_matrix(&work, &b_now, c.b.source_size, b_now.effect_time_at(c.b.t));
        let Some(fa) = pick(&c.a.faces, &ma, None) else {
            rows.push(json!({"seconds": secs(b_now.start), "from": a_id.0, "to": b_id.0, "status": "noFace", "side": "outgoing"}));
            continue;
        };
        let ea = apply(&ma, fa.eye_mid());
        let Some(fb) = pick(&c.b.faces, &mb, Some(ea)) else {
            rows.push(json!({"seconds": secs(b_now.start), "from": a_id.0, "to": b_id.0, "status": "noFace", "side": "incoming"}));
            continue;
        };
        let before = apply(&mb, fb.eye_mid());
        let a_eff = y_scale(&ma);
        let mode = if alternate {
            if a_eff > base * (1.0 + punch / 2.0) {
                ScaleMode::Percent(base / y_scale(&mb) * motion_values(&work, &b_now, b_now.effect_time_at(c.b.t)).1)
            } else {
                ScaleMode::PunchIn(punch)
            }
        } else {
            ScaleMode::Keep
        };
        if !alternate && (before.y - ea.y).abs() <= tol && o.axes == Axes::Y {
            rows.push(json!({"seconds": secs(b_now.start), "from": a_id.0, "to": b_id.0, "status": "alreadyAligned", "deltaY": r2(before.y - ea.y)}));
            continue;
        }
        let Some(pl) = plan(&work, &b_now, c.b.source_size, b_now.effect_time_at(c.b.t), fb.eye_mid(), fb.interocular(), ea, a_eff, 0.0, mode, o) else {
            rows.push(json!({"seconds": secs(b_now.start), "from": a_id.0, "to": b_id.0, "status": "unsolvable"}));
            continue;
        };
        if let Some((_, it)) = work.find_item_mut(b_id) {
            *it = with_plan(it, pl.position, pl.scale, pl.scale_width);
        }
        let mut row = plan_json(&pl, &b_now);
        row["seconds"] = json!(secs(b_now.start));
        row["from"] = json!(a_id.0);
        row["deltaYBefore"] = json!(r2(before.y - ea.y));
        row["status"] = json!(if pl.clamped { "clamped" } else { "aligned" });
        rows.push(row);
        plans.push((b_id, pl));
    }
    let keys = if plans.is_empty() { Vec::new() } else { apply_plans(s, "Align Jump Cuts", &plans)? };
    Ok(json!({
        "track": format!("V{}", track + 1), "count": cuts.len(), "changed": plans.len(), "cuts": rows,
        "keyframesRemovedFrom": keys, "detector": det.id(),
        "note": "one undo step; each incoming clip is solved against the outgoing clip's (already updated) transform, left to right",
    }))
}

fn face_model(_: &mut Session, _: &Value) -> Result<Value> {
    let dir = crate::transcript::models_dir();
    Ok(json!({
        "id": FACE_MODEL.id, "name": FACE_MODEL.name, "description": FACE_MODEL.description,
        "license": FACE_MODEL.license, "licenseUrl": FACE_MODEL.license_url, "author": FACE_MODEL.author, "source": FACE_MODEL.source,
        "size": FACE_MODEL.size(), "installed": models_installed(),
        "dir": dir.map(|d| filmcraft_speech::models::model_dir(&d, &FACE_MODEL).to_string_lossy().into_owned()),
        "canDownload": cfg!(feature = "speech-download"),
    }))
}

fn download_face_model(_: &mut Session, _: &Value) -> Result<Value> {
    download_models()?;
    Ok(json!({"model": FACE_MODEL.id, "installed": models_installed()}))
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    let spec = |id, label, params, enabled, run, journal| CommandSpec { id, label, menu: &[], shortcut: None, params, enabled, run, journal };
    vec![
        spec(
            "video.faces",
            "Find Faces",
            r#"{"clip":id?,"time|seconds|frame|timecode":t?,"source":"clip"|"program"?,"region":{"x":px,"y":px,"w":px,"h":px}?,"maxFaces":n=4,"minScore":0..1=0.5,"tiles":bool=true,"scale":f=1,"download":bool?}"#,
            has_seq,
            faces_cmd,
            false,
        ),
        spec(
            "video.alignEyes",
            "Align Eyes Across Cut",
            r#"{"from":clipA,"to":clipB,"mode":"match"|"punchIn"|"matchSize"="match","scale":percent?,"punchPercent":f=10,"axes":"y"|"xy"="y","clampBorders":bool=true,"autoScale":bool=false,"verify":bool=true,"region":{x,y,w,h}?,"download":bool?}"#,
            has_seq,
            align_eyes,
            true,
        ),
        spec(
            "video.auditJumpCuts",
            "Audit Jump-Cut Eye Lines",
            r#"{"track":"V1"?,"from":seconds?,"to":seconds?,"tolerance":px=2,"sameMedia":bool=false,"region":{x,y,w,h}?,"download":bool?}"#,
            has_seq,
            audit_jump_cuts,
            false,
        ),
        spec(
            "video.alignJumpCuts",
            "Align Jump-Cut Eye Lines",
            r#"{"track":"V1"?,"from":seconds?,"to":seconds?,"mode":"match"|"alternate"="match","punchPercent":f=10,"axes":"y"|"xy"="y","clampBorders":bool=true,"autoScale":bool=false,"tolerance":px=0.5,"sameMedia":bool=false,"region":{x,y,w,h}?,"download":bool?}"#,
            has_seq,
            align_jump_cuts,
            true,
        ),
        spec("video.faceModel", "Face Model Status", "{}", |_| Ok(()), face_model, false),
        spec("video.downloadFaceModel", "Download Face Model", "{}", can_download, download_face_model, true),
    ]
}
