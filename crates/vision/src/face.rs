//! Face and eye landmarks: MediaPipe BlazeFace (detector) followed by MediaPipe Face Mesh (468
//! landmarks), both run by the pure-Rust interpreter in [`crate::nn`].
//!
//! 1. **Detect.** The picture (or `region` of it) is letterboxed into the detector's square input
//!    and decoded against BlazeFace's SSD anchors; overlapping boxes merge by score-weighted
//!    non-maximum suppression. When nothing is found and the picture is large, overlapping square
//!    tiles of half the picture height are searched too (small faces, e.g. a webcam inset in a
//!    screen recording).
//! 2. **Refine.** Each face is cropped square (1.5 × its box), rotated so its eyes are level, and
//!    run through Face Mesh; the crop is then re-centred on the mesh and run once more (MediaPipe's
//!    tracking loop, done twice on one frame).
//! 3. **Eyes.** An eye's centre is the midpoint of its two corners (mesh points 33 / 133 and
//!    362 / 263): corners don't move when the eye blinks or looks around, so the centre measures
//!    where the head is, which is what an eye-line match across a cut needs.
//!
//! Without a mesh model (or when the mesh rejects the crop) the detector's own eye keypoints are
//! used, flagged `method: Detector` (coarser: a few pixels on a 1080p talking head).
//!
//! No face → an empty list. Every failure is an [`Error`], never a panic.

use serde::Serialize;

use crate::Error;
use crate::nn::{Net, Tensor};

/// A point in picture pixels (x right, y down; pixel centres at +0.5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
    fn mid(a: Point, b: Point) -> Point {
        Point::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
    }
}

/// An axis-aligned box in picture pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct BBox {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl BBox {
    fn iou(&self, o: &BBox) -> f64 {
        let ix = (self.x + self.w).min(o.x + o.w) - self.x.max(o.x);
        let iy = (self.y + self.h).min(o.y + o.h) - self.y.max(o.y);
        if ix <= 0.0 || iy <= 0.0 {
            return 0.0;
        }
        let i = ix * iy;
        let u = self.w * self.h + o.w * o.h - i;
        if u > 0.0 { i / u } else { 0.0 }
    }
    fn center(&self) -> Point {
        Point::new(self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
}

/// How a face's eye positions were found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Method {
    /// Face Mesh eye corners (precise).
    Mesh,
    /// The detector's eye keypoints only (coarse).
    Detector,
}

/// One face.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Face {
    /// Detector confidence (0–1).
    pub score: f32,
    /// Face Mesh's face-presence score (0–1), when the mesh ran.
    pub mesh_score: Option<f32>,
    /// Face box (from the mesh when it ran: the box around its 468 points).
    pub bbox: BBox,
    /// Centre of the eye on the **left of the picture** (the subject's right eye).
    pub left_eye: Point,
    /// Centre of the eye on the **right of the picture** (the subject's left eye).
    pub right_eye: Point,
    pub method: Method,
    /// The 468 mesh points (picture pixels), when the mesh ran.
    #[serde(skip)]
    pub mesh: Vec<Point>,
}

impl Face {
    /// Midpoint of the two eye centres.
    pub fn eye_mid(&self) -> Point {
        Point::mid(self.left_eye, self.right_eye)
    }
    /// Distance between the eye centres.
    pub fn interocular(&self) -> f64 {
        (self.right_eye.x - self.left_eye.x).hypot(self.right_eye.y - self.left_eye.y)
    }
    /// Angle of the eye line in degrees, clockwise on screen (positive: the right-hand eye is
    /// lower).
    pub fn eye_line_deg(&self) -> f64 {
        (self.right_eye.y - self.left_eye.y).atan2(self.right_eye.x - self.left_eye.x).to_degrees()
    }
}

/// An RGBA8 picture.
#[derive(Clone, Copy, Debug)]
pub struct Image<'a> {
    pub width: usize,
    pub height: usize,
    /// `width * height * 4` bytes, rows top to bottom.
    pub rgba: &'a [u8],
}

impl<'a> Image<'a> {
    pub fn new(width: usize, height: usize, rgba: &'a [u8]) -> Result<Self, Error> {
        let need = width.checked_mul(height).and_then(|n| n.checked_mul(4)).ok_or_else(|| Error::Input("picture too large".into()))?;
        if width == 0 || height == 0 || rgba.len() < need {
            return Err(Error::Input(format!("picture of {width}×{height} needs {need} bytes, got {}", rgba.len())));
        }
        Ok(Image { width, height, rgba })
    }

    fn px(&self, x: isize, y: isize) -> [f32; 3] {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return [0.0; 3];
        }
        let i = (y as usize * self.width + x as usize) * 4;
        match self.rgba.get(i..i + 3) {
            Some(p) => [f32::from(p[0]), f32::from(p[1]), f32::from(p[2])],
            None => [0.0; 3],
        }
    }

    /// Bilinear sample at picture coordinates (pixel centres at +0.5), black outside.
    fn bilinear(&self, x: f64, y: f64) -> [f32; 3] {
        let (fx, fy) = (x - 0.5, y - 0.5);
        if !(fx.is_finite() && fy.is_finite()) || fx < -2.0 || fy < -2.0 || fx > self.width as f64 + 1.0 || fy > self.height as f64 + 1.0 {
            return [0.0; 3];
        }
        let (x0, y0) = (fx.floor(), fy.floor());
        let (dx, dy) = ((fx - x0) as f32, (fy - y0) as f32);
        let (x0, y0) = (x0 as isize, y0 as isize);
        let (a, b, c, d) = (self.px(x0, y0), self.px(x0 + 1, y0), self.px(x0, y0 + 1), self.px(x0 + 1, y0 + 1));
        let mut o = [0f32; 3];
        for k in 0..3 {
            let top = a[k] + (b[k] - a[k]) * dx;
            let bot = c[k] + (d[k] - c[k]) * dx;
            o[k] = top + (bot - top) * dy;
        }
        o
    }
}

/// A square crop of the picture: `center`, side `size` (picture px), rotated so its x axis runs
/// along `axis` (a unit vector; (1, 0) = upright).
#[derive(Clone, Copy, Debug)]
struct Crop {
    center: Point,
    size: f64,
    axis: (f64, f64),
}

impl Crop {
    /// Crop coordinates (0–1 across the crop) → picture pixels.
    fn at(&self, s: f64, t: f64) -> Point {
        let (u, v) = ((s - 0.5) * self.size, (t - 0.5) * self.size);
        let (ax, ay) = self.axis;
        Point::new(self.center.x + u * ax - v * ay, self.center.y + u * ay + v * ax)
    }

    /// Sample the crop into an `n`×`n` NHWC tensor, values mapped from 0–255 to `lo`–`hi`. Each
    /// output pixel averages up to 8×8 bilinear taps when the crop is shrunk (no aliasing).
    fn tensor(&self, img: &Image, n: usize, lo: f32, hi: f32) -> Result<Tensor, Error> {
        let px = self.size / n as f64;
        let taps = (px.ceil() as usize).clamp(1, 8);
        let mut data = Vec::with_capacity(n * n * 3);
        let k = (hi - lo) / 255.0;
        for j in 0..n {
            for i in 0..n {
                let mut acc = [0f32; 3];
                for tj in 0..taps {
                    for ti in 0..taps {
                        let s = (i as f64 + (ti as f64 + 0.5) / taps as f64) / n as f64;
                        let t = (j as f64 + (tj as f64 + 0.5) / taps as f64) / n as f64;
                        let p = self.at(s, t);
                        let c = img.bilinear(p.x, p.y);
                        for q in 0..3 {
                            acc[q] += c[q];
                        }
                    }
                }
                let w = 1.0 / (taps * taps) as f32;
                for a in acc {
                    data.push(lo + a * w * k);
                }
            }
        }
        Tensor::new(vec![1, n, n, 3], data)
    }
}

/// Which BlazeFace detector a model file is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectorKind {
    /// `face_detection_short_range.tflite`: 128 px input, faces within ~2 m of a selfie camera.
    ShortRange,
    /// `face_detection_full_range.tflite`: 192 px input, faces up to ~5 m (smaller in frame).
    FullRange,
}

impl DetectorKind {
    fn input(self) -> usize {
        match self {
            DetectorKind::ShortRange => 128,
            DetectorKind::FullRange => 192,
        }
    }
    /// SSD anchor centres (normalised), in output order. Both models use fixed-size anchors.
    fn anchors(self) -> Vec<(f32, f32)> {
        // (grid size, anchors per cell)
        let layers: &[(usize, usize)] = match self {
            // strides 8 (2 anchors: aspect 1 + interpolated) and 16 ×3 layers (6 anchors)
            DetectorKind::ShortRange => &[(16, 2), (8, 6)],
            // stride 4, one anchor
            DetectorKind::FullRange => &[(48, 1)],
        };
        let mut v = Vec::new();
        for (g, per) in layers {
            for y in 0..*g {
                for x in 0..*g {
                    for _ in 0..*per {
                        v.push(((x as f32 + 0.5) / *g as f32, (y as f32 + 0.5) / *g as f32));
                    }
                }
            }
        }
        v
    }
}

/// A raw detection in picture pixels: box, 6 keypoints (subject's right eye, left eye, nose tip,
/// mouth, right ear, left ear), score.
#[derive(Clone, Debug)]
struct Detection {
    bbox: BBox,
    kps: [Point; 6],
    score: f32,
}

/// Options for [`FaceModels::detect`].
#[derive(Clone, Copy, Debug)]
pub struct DetectOptions {
    /// Minimum detector confidence (default 0.5).
    pub min_score: f32,
    /// Keep at most this many faces, largest first (default 4).
    pub max_faces: usize,
    /// Search only this part of the picture (picture px).
    pub region: Option<BBox>,
    /// Also search tiles when the whole picture finds nothing (default true).
    pub tiles: bool,
}

impl Default for DetectOptions {
    fn default() -> Self {
        DetectOptions { min_score: 0.5, max_faces: 4, region: None, tiles: true }
    }
}

/// The loaded models.
#[derive(Debug)]
pub struct FaceModels {
    detector: Net,
    kind: DetectorKind,
    mesh: Option<Net>,
}

/// A detection Face Mesh rejects is kept (detector eyes only) at this confidence or more.
const STRONG_DETECTION: f32 = 0.75;
/// Face Mesh input size.
const MESH_IN: usize = 192;
/// Mesh points of the eye corners: (subject's right eye: outer, inner), (left eye: inner, outer).
const EYE_CORNERS: [usize; 4] = [33, 133, 362, 263];

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v.clamp(-100.0, 100.0)).exp())
}

impl FaceModels {
    /// Load a BlazeFace detector and (optionally) Face Mesh from their `.tflite` bytes.
    pub fn load(detector: &[u8], kind: DetectorKind, mesh: Option<&[u8]>) -> Result<Self, Error> {
        let det = Net::load(detector)?;
        let n = kind.input();
        if det.input_shape(0) != Some(&[1, n, n, 3][..]) {
            return Err(Error::Model(format!("the detector takes {:?}, expected [1,{n},{n},3]", det.input_shape(0))));
        }
        let mesh = match mesh {
            Some(b) => {
                let m = Net::load(b)?;
                if m.input_shape(0) != Some(&[1, MESH_IN, MESH_IN, 3][..]) {
                    return Err(Error::Model("the face mesh model must take [1,192,192,3]".into()));
                }
                Some(m)
            }
            None => None,
        };
        Ok(FaceModels { detector: det, kind, mesh })
    }

    /// Faces in `img`, largest first.
    pub fn detect(&self, img: &Image, opts: &DetectOptions) -> Result<Vec<Face>, Error> {
        let area = match opts.region {
            Some(r) => {
                let x0 = r.x.max(0.0);
                let y0 = r.y.max(0.0);
                let x1 = (r.x + r.w).min(img.width as f64);
                let y1 = (r.y + r.h).min(img.height as f64);
                if !(x1 - x0 >= 8.0 && y1 - y0 >= 8.0) {
                    return Err(Error::Input("the region is empty or outside the picture".into()));
                }
                BBox { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
            }
            None => BBox { x: 0.0, y: 0.0, w: img.width as f64, h: img.height as f64 },
        };
        let mut dets = self.detect_in(img, area, opts.min_score)?;
        if dets.is_empty() && opts.tiles {
            // small faces: overlapping square tiles of half the area's short side
            let side = (area.w.min(area.h) / 2.0).max(64.0);
            if side * 1.5 < area.w.max(area.h) || side * 2.0 <= area.w.min(area.h) + 1.0 {
                let step = side / 2.0;
                let nx = (((area.w - side) / step).ceil().max(0.0) as usize + 1).min(16);
                let ny = (((area.h - side) / step).ceil().max(0.0) as usize + 1).min(16);
                let mut all = Vec::new();
                for j in 0..ny {
                    for i in 0..nx {
                        let x = (area.x + i as f64 * step).min(area.x + area.w - side).max(area.x);
                        let y = (area.y + j as f64 * step).min(area.y + area.h - side).max(area.y);
                        all.extend(self.raw_detect(img, BBox { x, y, w: side, h: side }, opts.min_score)?);
                    }
                }
                dets = weighted_nms(all, 0.3);
            }
        }
        // refine the strongest candidates (several detections of one face converge on it)
        let mut faces: Vec<Face> = Vec::new();
        for d in dets.iter().take(opts.max_faces.clamp(1, 64).saturating_mul(4)) {
            faces.extend(self.refine(img, d)?);
        }
        Ok(rank_faces(faces, opts.max_faces.clamp(1, 64)))
    }

    fn detect_in(&self, img: &Image, area: BBox, min_score: f32) -> Result<Vec<Detection>, Error> {
        Ok(weighted_nms(self.raw_detect(img, area, min_score)?, 0.3))
    }

    /// Run the detector on `area` letterboxed into its square input.
    fn raw_detect(&self, img: &Image, area: BBox, min_score: f32) -> Result<Vec<Detection>, Error> {
        let n = self.kind.input();
        let side = area.w.max(area.h);
        let crop = Crop { center: area.center(), size: side, axis: (1.0, 0.0) };
        let input = crop.tensor(img, n, -1.0, 1.0)?;
        let out = self.detector.run(input)?;
        // outputs: regressors [1, anchors, 16] and scores [1, anchors, 1], in either order
        let (mut regs, mut scores) = (None, None);
        for t in &out {
            match t.shape().last() {
                Some(16) => regs = Some(t.data()),
                Some(1) => scores = Some(t.data()),
                _ => {}
            }
        }
        let (Some(regs), Some(scores)) = (regs, scores) else { return Err(Error::Model("unexpected detector outputs".into())) };
        let anchors = self.kind.anchors();
        if scores.len() != anchors.len() || regs.len() != anchors.len() * 16 {
            return Err(Error::Model(format!("the detector gives {} boxes, expected {}", scores.len(), anchors.len())));
        }
        let scale = n as f32;
        let mut v = Vec::new();
        let to_px = |nx: f32, ny: f32| -> Point { Point::new(area.center().x + (f64::from(nx) - 0.5) * side, area.center().y + (f64::from(ny) - 0.5) * side) };
        for (k, (ax, ay)) in anchors.iter().enumerate() {
            let s = sigmoid(scores[k]);
            if s.is_nan() || s < min_score {
                continue;
            }
            let r = &regs[k * 16..k * 16 + 16];
            let (cx, cy) = (r[0] / scale + ax, r[1] / scale + ay);
            let (w, h) = (r[2] / scale, r[3] / scale);
            if !(w > 0.0 && h > 0.0 && cx.is_finite() && cy.is_finite()) {
                continue;
            }
            let tl = to_px(cx - w / 2.0, cy - h / 2.0);
            let mut kps = [Point::default(); 6];
            for (j, p) in kps.iter_mut().enumerate() {
                *p = to_px(r[4 + 2 * j] / scale + ax, r[5 + 2 * j] / scale + ay);
            }
            v.push(Detection { bbox: BBox { x: tl.x, y: tl.y, w: f64::from(w) * side, h: f64::from(h) * side }, kps, score: s });
        }
        Ok(v)
    }

    /// Mesh refinement of one detection (falls back to the detector's keypoints; `None`: the mesh
    /// rejected a weak detection).
    fn refine(&self, img: &Image, d: &Detection) -> Result<Option<Face>, Error> {
        let coarse =
            Face { score: d.score, mesh_score: None, bbox: d.bbox, left_eye: d.kps[0], right_eye: d.kps[1], method: Method::Detector, mesh: Vec::new() };
        let Some(mesh) = &self.mesh else { return Ok(Some(coarse)) };
        let axis = unit(d.kps[0], d.kps[1]);
        let mut crop = Crop { center: d.bbox.center(), size: d.bbox.w.max(d.bbox.h) * 1.5, axis };
        let mut best: Option<(Vec<Point>, f32)> = None;
        for _ in 0..2 {
            if !(crop.size >= 4.0 && crop.size.is_finite()) {
                break;
            }
            let out = mesh.run(crop.tensor(img, MESH_IN, 0.0, 1.0)?)?;
            let pts = out.iter().find(|t| t.data().len() == 468 * 3).ok_or_else(|| Error::Model("unexpected face mesh outputs".into()))?;
            let presence = out.iter().find(|t| t.data().len() == 1).and_then(|t| t.data().first().copied()).map(sigmoid).unwrap_or(0.0);
            let p: Vec<Point> = pts.data().chunks_exact(3).map(|c| crop.at(f64::from(c[0]) / MESH_IN as f64, f64::from(c[1]) / MESH_IN as f64)).collect();
            if presence < 0.5 {
                break;
            }
            // the next crop: around the mesh, level with its eye corners
            let (x0, x1) = p.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), q| (a.min(q.x), b.max(q.x)));
            let (y0, y1) = p.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), q| (a.min(q.y), b.max(q.y)));
            if let (Some(a), Some(b)) = (p.get(33), p.get(263)) {
                crop = Crop { center: Point::new((x0 + x1) / 2.0, (y0 + y1) / 2.0), size: (x1 - x0).max(y1 - y0) * 1.5, axis: unit(*a, *b) };
            }
            best = Some((p, presence));
        }
        // the mesh found no face in the crop: keep only a confident detection (a profile)
        let Some((p, presence)) = best else { return Ok(if d.score >= STRONG_DETECTION { Some(coarse) } else { None }) };
        let get = |i: usize| p.get(i).copied().unwrap_or_default();
        let [a, b, c, e] = EYE_CORNERS.map(get);
        let (x0, x1) = p.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), q| (lo.min(q.x), hi.max(q.x)));
        let (y0, y1) = p.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), q| (lo.min(q.y), hi.max(q.y)));
        Ok(Some(Face {
            score: d.score,
            mesh_score: Some(presence),
            bbox: BBox { x: x0, y: y0, w: x1 - x0, h: y1 - y0 },
            left_eye: Point::mid(a, b),
            right_eye: Point::mid(c, e),
            method: Method::Mesh,
            mesh: p,
        }))
    }
}

/// Mesh-confirmed faces first, then by size; a face whose centre lies inside a face already kept
/// is the same face found twice and is dropped.
fn rank_faces(mut faces: Vec<Face>, max: usize) -> Vec<Face> {
    let key = |f: &Face| (f.method == Method::Mesh, f.bbox.w * f.bbox.h);
    faces.sort_by(|a, b| {
        let (ka, kb) = (key(a), key(b));
        kb.0.cmp(&ka.0).then(kb.1.total_cmp(&ka.1))
    });
    let mut kept: Vec<Face> = Vec::new();
    for f in faces {
        let c = f.bbox.center();
        let dup = kept.iter().any(|k| {
            let inside = |b: &BBox, p: Point| p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h;
            inside(&k.bbox, c) || inside(&f.bbox, k.bbox.center()) || k.bbox.iou(&f.bbox) > 0.3
        });
        if !dup && kept.len() < max {
            kept.push(f);
        }
    }
    kept
}

/// Unit vector from `a` to `b` ((1, 0) when they coincide).
fn unit(a: Point, b: Point) -> (f64, f64) {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let l = dx.hypot(dy);
    if l > 1e-6 && l.is_finite() { (dx / l, dy / l) } else { (1.0, 0.0) }
}

/// Score-weighted non-maximum suppression (MediaPipe's "weighted" mode): every box overlapping
/// the best remaining one by more than `iou` is merged into it, weighted by score.
fn weighted_nms(mut v: Vec<Detection>, iou: f64) -> Vec<Detection> {
    v.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut out = Vec::new();
    while !v.is_empty() {
        let top = v[0].bbox;
        // the best box always joins its own group (even when degenerate)
        let (group, rest): (Vec<(usize, Detection)>, Vec<(usize, Detection)>) = v.into_iter().enumerate().partition(|(i, d)| *i == 0 || d.bbox.iou(&top) > iou);
        v = rest.into_iter().map(|(_, d)| d).collect();
        let group: Vec<Detection> = group.into_iter().map(|(_, d)| d).collect();
        let Some(mut m) = group.first().cloned() else { break };
        let total: f64 = group.iter().map(|d| f64::from(d.score)).sum();
        if total > 0.0 {
            let w = |d: &Detection| f64::from(d.score) / total;
            m.bbox = BBox {
                x: group.iter().map(|d| d.bbox.x * w(d)).sum(),
                y: group.iter().map(|d| d.bbox.y * w(d)).sum(),
                w: group.iter().map(|d| d.bbox.w * w(d)).sum(),
                h: group.iter().map(|d| d.bbox.h * w(d)).sum(),
            };
            for k in 0..6 {
                m.kps[k] = Point::new(group.iter().map(|d| d.kps[k].x * w(d)).sum(), group.iter().map(|d| d.kps[k].y * w(d)).sum());
            }
        }
        out.push(m);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors_match_the_models() {
        assert_eq!(DetectorKind::ShortRange.anchors().len(), 896);
        assert_eq!(DetectorKind::FullRange.anchors().len(), 2304);
        let a = DetectorKind::FullRange.anchors();
        assert_eq!(a[0], (0.5 / 48.0, 0.5 / 48.0));
        assert_eq!(a[49], (1.5 / 48.0, 1.5 / 48.0));
    }

    #[test]
    fn nms_merges_overlaps_and_keeps_separate_faces() {
        let d = |x: f64, s: f32| Detection { bbox: BBox { x, y: 0.0, w: 10.0, h: 10.0 }, kps: [Point::new(x, 0.0); 6], score: s };
        let out = weighted_nms(vec![d(0.0, 0.9), d(1.0, 0.9), d(50.0, 0.6)], 0.3);
        assert_eq!(out.len(), 2);
        assert!((out[0].bbox.x - 0.5).abs() < 1e-9);
        assert_eq!(out[1].bbox.x, 50.0);
        // zero-area boxes don't loop forever
        let z = |s: f32| Detection { bbox: BBox::default(), kps: [Point::default(); 6], score: s };
        assert_eq!(weighted_nms(vec![z(0.9), z(0.8)], 0.3).len(), 2);
    }

    #[test]
    fn duplicates_collapse_and_mesh_faces_rank_first() {
        let f = |x: f64, w: f64, m: Method| Face {
            score: 0.9,
            mesh_score: None,
            bbox: BBox { x, y: 0.0, w, h: w },
            left_eye: Point::default(),
            right_eye: Point::default(),
            method: m,
            mesh: Vec::new(),
        };
        let out = rank_faces(vec![f(0.0, 200.0, Method::Detector), f(10.0, 150.0, Method::Mesh), f(500.0, 50.0, Method::Mesh)], 4);
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].method, out[0].bbox.x), (Method::Mesh, 10.0));
        assert_eq!(out[1].bbox.x, 500.0);
        assert_eq!(rank_faces(vec![f(0.0, 1.0, Method::Mesh), f(9.0, 1.0, Method::Mesh)], 1).len(), 1);
    }

    #[test]
    fn crop_maps_and_rotates() {
        let c = Crop { center: Point::new(100.0, 50.0), size: 20.0, axis: (1.0, 0.0) };
        assert_eq!(c.at(0.0, 0.0), Point::new(90.0, 40.0));
        assert_eq!(c.at(1.0, 1.0), Point::new(110.0, 60.0));
        // axis pointing down: crop x runs down the picture, crop y runs to the left
        let r = Crop { axis: (0.0, 1.0), ..c };
        let p = r.at(1.0, 0.5);
        assert!((p.x - 100.0).abs() < 1e-9 && (p.y - 60.0).abs() < 1e-9);
        let p = r.at(0.5, 1.0);
        assert!((p.x - 90.0).abs() < 1e-9 && (p.y - 50.0).abs() < 1e-9);
    }

    #[test]
    fn image_rejects_short_buffers() {
        assert!(Image::new(10, 10, &[0; 399]).is_err());
        assert!(Image::new(0, 10, &[]).is_err());
        assert!(Image::new(usize::MAX, 2, &[]).is_err());
        let px = [255u8; 16];
        let img = Image::new(2, 2, &px).unwrap();
        assert_eq!(img.bilinear(1.0, 1.0), [255.0; 3]);
        assert_eq!(img.bilinear(f64::NAN, 1.0), [0.0; 3]);
        assert_eq!(img.bilinear(-1e300, 1e300), [0.0; 3]);
    }

    #[test]
    fn face_geometry() {
        let f = Face {
            score: 1.0,
            mesh_score: None,
            bbox: BBox::default(),
            left_eye: Point::new(0.0, 0.0),
            right_eye: Point::new(10.0, 10.0),
            method: Method::Mesh,
            mesh: Vec::new(),
        };
        assert_eq!(f.eye_mid(), Point::new(5.0, 5.0));
        assert!((f.eye_line_deg() - 45.0).abs() < 1e-9);
        assert!((f.interocular() - 200f64.sqrt()).abs() < 1e-9);
    }
}
