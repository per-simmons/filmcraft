//! Audio clip waveforms, drawn as Premiere Pro draws them: every channel in its own lane, filling
//! the clip below its name, rectified (one-sided, rising from the bottom of the lane) by default
//! or centred and two-sided (Timeline Display Settings ▸ Rectified Audio Waveforms), on a
//! logarithmic (dB) scale by default so quiet speech stays readable (View ▸ Dynamic Audio
//! Waveforms / Logarithmic Waveform Scaling; off = linear amplitude). Waveforms scale with the
//! track's height.
//!
//! Peaks are computed once per media item (in the background, see `timeline::request_peaks`):
//! a min / max pair per [`BUCKET`] samples at 48 kHz per channel, plus a pyramid of coarser levels
//! (each [`FAN`] times coarser). Drawing reads one min / max per device pixel column from the
//! level that fits the zoom, so a zoomed-out clip is a true per-pixel envelope (no sparse,
//! aliased look) for a few lookups per column, and a zoomed-in one is interpolated between peaks
//! (smooth, no stair steps). Nothing is recomputed per frame.

use egui::{Color32, Painter, Rect, pos2};

/// Samples (at 48 kHz) per base peak.
pub const BUCKET: usize = 256;
/// Channels kept per item (a 7.1 file has 8).
pub const MAX_CHANNELS: usize = 8;
/// Each pyramid level is this many times coarser than the one below.
const FAN: usize = 4;
/// Floor of the logarithmic scale: −48 dBFS draws at the bottom, 0 dBFS fills the lane.
pub const DB_FLOOR: f32 = -48.0;
/// Lanes smaller than this (points) are merged into one (a minimized track).
const MIN_LANE_H: f32 = 7.0;

/// Waveform peaks of one item: per channel, min / max pairs at every pyramid level.
#[derive(Debug, Default)]
pub struct Peaks {
    /// `channels[c][level][i]`; level 0 is [`BUCKET`] samples a pair.
    channels: Vec<Vec<Vec<(f32, f32)>>>,
    max_abs: f32,
}

impl Peaks {
    /// Peaks from per-channel base pairs (at most [`MAX_CHANNELS`] are kept).
    pub fn from_channels(base: Vec<Vec<(f32, f32)>>) -> Peaks {
        let mut max_abs = 0f32;
        let channels = base
            .into_iter()
            .take(MAX_CHANNELS)
            .map(|b| {
                for (lo, hi) in &b {
                    if lo.is_finite() && hi.is_finite() {
                        max_abs = max_abs.max(lo.abs()).max(hi.abs());
                    }
                }
                let mut levels = vec![b];
                while let Some(last) = levels.last().filter(|l| l.len() > FAN) {
                    let next: Vec<(f32, f32)> = last.chunks(FAN).map(|c| c.iter().fold((0f32, 0f32), |(l, h), (a, b)| (l.min(*a), h.max(*b)))).collect();
                    levels.push(next);
                }
                levels
            })
            .collect();
        Peaks { channels, max_abs }
    }

    /// Peaks of one channel (for tests and simple callers).
    pub fn mono(base: Vec<(f32, f32)>) -> Peaks {
        Peaks::from_channels(vec![base])
    }

    pub fn channels(&self) -> usize {
        self.channels.len()
    }

    /// The base peaks of channel `ch` (empty when there is no such channel).
    pub fn base(&self, ch: usize) -> &[(f32, f32)] {
        self.channels.get(ch).and_then(|c| c.first()).map_or(&[], Vec::as_slice)
    }

    /// Base peaks per channel.
    pub fn len(&self) -> usize {
        self.channels.first().and_then(|c| c.first()).map_or(0, Vec::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The largest magnitude anywhere (for the display normalisation).
    pub fn max_abs(&self) -> f32 {
        self.max_abs
    }

    /// (min, max) of channel `ch` over base peaks `a..b` (fractional positions, in either order).
    /// Narrower than one peak (zoomed in), the envelope is interpolated between neighbouring peaks.
    /// Outside the data, or for a missing channel, (0, 0).
    pub fn range(&self, ch: usize, a: f64, b: f64) -> (f32, f32) {
        let Some(levels) = self.channels.get(ch) else { return (0.0, 0.0) };
        let Some(base) = levels.first() else { return (0.0, 0.0) };
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        let n = base.len() as f64;
        if !a.is_finite() || !b.is_finite() || b < 0.0 || a >= n || n == 0.0 {
            return (0.0, 0.0);
        }
        let span = b - a;
        if span <= 1.0 {
            // linear interpolation between the centres of the neighbouring peaks
            let c = ((a + b) * 0.5 - 0.5).clamp(0.0, n - 1.0);
            let i = c.floor() as usize;
            let f = (c - i as f64) as f32;
            let p0 = base.get(i).copied().unwrap_or_default();
            let p1 = base.get(i + 1).copied().unwrap_or(p0);
            return (p0.0 + (p1.0 - p0.0) * f, p0.1 + (p1.1 - p0.1) * f);
        }
        // the coarsest level whose peaks are at most a quarter of the span: a handful of lookups
        let mut level = 0usize;
        let mut size = 1f64;
        while level + 1 < levels.len() && size * (FAN as f64) * 4.0 <= span {
            level += 1;
            size *= FAN as f64;
        }
        let Some(l) = levels.get(level) else { return (0.0, 0.0) };
        let i0 = (a.max(0.0) / size).floor() as usize;
        let i1 = ((b / size).ceil() as usize).min(l.len()).max(i0 + 1);
        l.get(i0..i1.min(l.len())).unwrap_or(&[]).iter().fold((0f32, 0f32), |(lo, hi), (x, y)| (lo.min(*x), hi.max(*y)))
    }

    /// (min, max) over every channel (a one-lane overview).
    pub fn range_all(&self, a: f64, b: f64) -> (f32, f32) {
        (0..self.channels()).map(|c| self.range(c, a, b)).fold((0f32, 0f32), |(l, h), (x, y)| (l.min(x), h.max(y)))
    }
}

/// Min / max pairs of `BUCKET`-sample (or `bucket`-sample) blocks of a channel.
pub fn block_peaks(samples: &[f32], bucket: usize, out: &mut Vec<(f32, f32)>) {
    for c in samples.chunks(bucket.max(1)) {
        out.push(c.iter().filter(|v| v.is_finite()).fold((0f32, 0f32), |(l, h), v| (l.min(*v), h.max(*v))));
    }
}

/// Height (0…1 of the lane, or of the half-lane when centred) for a magnitude `amp` (gain
/// applied). Logarithmic: [`DB_FLOOR`] → 0, 0 dBFS → 1, so −24 dBFS speech fills half the lane.
/// Linear: the amplitude itself. Always within 0…1 (NaN → 0).
pub fn amp_height(amp: f32, log: bool) -> f32 {
    let a = amp.abs();
    if a.is_nan() || a <= 0.0 {
        return 0.0;
    }
    if !log {
        return a.min(1.0);
    }
    let db = 20.0 * a.log10();
    ((db - DB_FLOOR) / -DB_FLOOR).clamp(0.0, 1.0)
}

/// How to draw a clip's waveform.
#[derive(Clone, Copy, Debug)]
pub struct Style {
    pub rectified: bool,
    pub log: bool,
    /// Display gain (clip gain and normalisation).
    pub gain: f32,
    pub color: Color32,
    /// Device pixels per point (one column per device pixel).
    pub pixels_per_point: f32,
}

/// The lanes of `area` for `channels` channels: one each (1 point apart), or one for all when
/// they would be too thin. Returns the lane rects and whether the lanes are merged.
pub fn lanes(area: Rect, channels: usize) -> (Vec<Rect>, bool) {
    let n = channels.max(1);
    let gap = 1.0;
    let h = (area.height() - gap * (n - 1) as f32) / n as f32;
    if n == 1 || h < MIN_LANE_H {
        return (vec![area], n > 1);
    }
    let lanes = (0..n)
        .map(|i| {
            let y = area.min.y + i as f32 * (h + gap);
            Rect::from_min_max(pos2(area.min.x, y), pos2(area.max.x, y + h))
        })
        .collect();
    (lanes, false)
}

/// Paint the waveform of `peaks` into `area`, columns clipped to `visible`. `pos_at(x)` is the
/// base peak position (fractional) shown at screen x. Returns the lanes drawn.
pub fn paint(p: &Painter, area: Rect, visible: Rect, peaks: &Peaks, pos_at: impl Fn(f32) -> f64, s: Style) -> Vec<Rect> {
    let (lanes, merged) = lanes(area, peaks.channels());
    let vis = visible.intersect(area);
    if vis.width() <= 0.0 || vis.height() <= 0.0 || peaks.is_empty() {
        return lanes;
    }
    let dx = 1.0 / s.pixels_per_point.clamp(0.5, 4.0);
    let n = ((vis.width() / dx).ceil() as usize).min(64_000);
    // the source position at each column edge
    let edges: Vec<f64> = (0..=n + 1).map(|i| pos_at(vis.min.x + (i as f32 - 0.5) * dx)).collect();
    let height = |v: f32| amp_height(v * s.gain, s.log);
    let mut mesh = egui::Mesh::default();
    for (ci, lane) in lanes.iter().enumerate() {
        if lane.intersect(vis).height() <= 0.0 {
            continue;
        }
        let mid = lane.center().y;
        let half = lane.height() * 0.5;
        if !s.rectified {
            // the zero line of a centred waveform
            mesh.add_colored_rect(Rect::from_min_max(pos2(vis.min.x, mid - 0.25), pos2(vis.max.x, mid + 0.25)), s.color.gamma_multiply(0.45));
        }
        let mut prev: Option<(f32, f32, f32)> = None;
        for i in 0..=n {
            let x = (vis.min.x + i as f32 * dx).min(vis.max.x);
            let (Some(a), Some(b)) = (edges.get(i), edges.get(i + 1)) else { break };
            let (lo, hi) = if merged { peaks.range_all(*a, *b) } else { peaks.range(ci, *a, *b) };
            let (top, bottom) = if s.rectified {
                let m = lo.abs().max(hi.abs());
                (lane.max.y - height(m) * lane.height(), lane.max.y)
            } else {
                let up = if hi > 0.0 { height(hi) } else { -height(hi) };
                let down = if lo < 0.0 { height(lo) } else { -height(lo) };
                ((mid - up * half).min(mid), (mid + down * half).max(mid))
            };
            if let Some((px, pt, pb)) = prev
                && (pb - pt > 0.05 || bottom - top > 0.05)
            {
                let base = mesh.vertices.len() as u32;
                for (vx, vy) in [(px, pt), (px, pb), (x, top), (x, bottom)] {
                    mesh.colored_vertex(pos2(vx, vy), s.color);
                }
                mesh.add_triangle(base, base + 1, base + 2);
                mesh.add_triangle(base + 1, base + 3, base + 2);
            }
            prev = Some((x, top, bottom));
        }
    }
    p.add(mesh);
    lanes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db(x: f32) -> f32 {
        10f32.powf(x / 20.0)
    }

    #[test]
    fn the_log_scale_maps_decibels_linearly_onto_the_lane() {
        assert_eq!(amp_height(1.0, true), 1.0, "0 dBFS fills the lane");
        assert!((amp_height(db(-24.0), true) - 0.5).abs() < 1e-4, "−24 dBFS: half");
        assert!((amp_height(db(-12.0), true) - 0.75).abs() < 1e-4);
        assert_eq!(amp_height(db(-48.0), true), 0.0, "the floor");
        assert_eq!(amp_height(db(-70.0), true), 0.0, "below the floor: nothing");
        // quiet speech is readable on the log scale, a sliver on the linear one
        assert!(amp_height(db(-30.0), true) > 0.35);
        assert!(amp_height(db(-30.0), false) < 0.04);
        // sign does not matter, overs clamp, junk is harmless
        assert_eq!(amp_height(-0.5, true), amp_height(0.5, true));
        assert_eq!(amp_height(4.0, true), 1.0);
        assert_eq!(amp_height(4.0, false), 1.0);
        for junk in [f32::NAN, 0.0, -0.0, f32::NEG_INFINITY] {
            assert!(amp_height(junk, true) == 0.0 || junk == f32::NEG_INFINITY);
        }
        assert_eq!(amp_height(f32::INFINITY, true), 1.0);
        // monotonic
        let mut last = -1.0;
        for i in 0..=100 {
            let h = amp_height(i as f32 / 100.0, true);
            assert!(h >= last);
            last = h;
        }
    }

    #[test]
    fn zoomed_out_columns_see_every_peak_through_the_pyramid() {
        // a single click in ten minutes of silence: every zoom level must still show it
        let n = 48_000 * 600 / BUCKET;
        let mut base = vec![(0.0f32, 0.0f32); n];
        base[77_777] = (-0.9, 0.8);
        let p = Peaks::mono(base);
        assert_eq!(p.len(), n);
        assert!((p.max_abs() - 0.9).abs() < 1e-6);
        for span in [1.5, 7.0, 64.0, 1000.0, 50_000.0] {
            let a = 77_777.0 - span * 0.3;
            assert_eq!(p.range(0, a, a + span), (-0.9, 0.8), "span {span}");
            // a column next to it (well past) does not pick it up
            let far = 77_778.0 + span * 2.0;
            assert_eq!(p.range(0, far, far + span), (0.0, 0.0), "span {span}");
        }
        assert_eq!(p.range(0, 0.0, n as f64), (-0.9, 0.8));
    }

    #[test]
    fn zoomed_in_columns_interpolate_between_peaks() {
        let p = Peaks::mono(vec![(-0.2, 0.2), (-0.6, 0.6)]);
        // centres at 0.5 and 1.5: halfway between them, halfway between the values
        let (lo, hi) = p.range(0, 0.95, 1.05);
        assert!((hi - 0.4).abs() < 1e-5 && (lo + 0.4).abs() < 1e-5);
        assert_eq!(p.range(0, 0.4, 0.6), (-0.2, 0.2));
        // outside the data, reversed, non-finite, missing channel: harmless
        assert_eq!(p.range(0, 5.0, 6.0), (0.0, 0.0));
        assert_eq!(p.range(0, 1.05, 0.95), p.range(0, 0.95, 1.05));
        assert_eq!(p.range(0, f64::NAN, 1.0), (0.0, 0.0));
        assert_eq!(p.range(3, 0.0, 1.0), (0.0, 0.0));
        assert_eq!(Peaks::default().range(0, 0.0, 1.0), (0.0, 0.0));
    }

    #[test]
    fn stereo_gets_a_lane_per_channel_until_the_track_is_too_small() {
        let p = Peaks::from_channels(vec![vec![(-0.1, 0.1); 8], vec![(-0.7, 0.7); 8], vec![]]);
        assert_eq!(p.channels(), 3);
        let (l, merged) = lanes(Rect::from_min_size(pos2(0.0, 0.0), egui::vec2(100.0, 41.0)), 2);
        assert_eq!((l.len(), merged), (2, false));
        assert_eq!((l[0].height(), l[1].min.y), (20.0, 21.0));
        let (l, merged) = lanes(Rect::from_min_size(pos2(0.0, 0.0), egui::vec2(100.0, 10.0)), 2);
        assert_eq!((l.len(), merged), (1, true));
        assert_eq!(p.range_all(0.0, 8.0), (-0.7, 0.7));
        let too_many = Peaks::from_channels(vec![vec![(0.0, 0.0)]; 20]);
        assert_eq!(too_many.channels(), MAX_CHANNELS);
    }

    #[test]
    fn block_peaks_skip_non_finite_samples() {
        let mut out = Vec::new();
        block_peaks(&[0.1, f32::NAN, -0.3, 0.2, f32::INFINITY, 0.5], 3, &mut out);
        assert_eq!(out, vec![(-0.3, 0.1), (0.0, 0.5)]);
        block_peaks(&[0.1], 0, &mut out);
        assert_eq!(out.len(), 3);
    }
}
