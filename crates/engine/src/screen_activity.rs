//! `video.screenActivity`: where a screen recording is changing, measured instead of looked at.
//!
//! For one video clip (default: the topmost enabled one at `from`), every `stepMs` (100 ms) of a
//! stretch of the sequence, the clip's source frame is decoded at reduced size, turned grey, cropped
//! to `region` (source pixels: the demo window, leaving out the webcam and terminal panes) and
//! compared with the previous step: `fraction` is the share of pixels whose grey level moved by more
//! than `threshold` (12 of 255), `mean` the mean absolute change. Steps at or over `activeFraction`
//! (0.0003: a pen stroke or a cursor moving is a few dozen pixels) are active; active steps closer
//! than `mergeMs` (300 ms) join into one range, and the end of each range is where the screen
//! settles. This is what the demo-breathing and spoken-app-command holds need ("keep the footage
//! until the app has visibly acted, end 0.2–0.3 s after it settles"), without screenshots or ffmpeg.
//!
//! Synchronous and read-only; at most 1 200 steps per call.

use serde_json::{Value, json};

use filmcraft_project::{ClipId, TrackKind};
use filmcraft_time::{Tick, TimeRange};

use crate::commands::{CommandSpec, bad, f64_p, has_seq, time_p, u64_p};
use crate::{EngineError, Result, Session};

const MAX_STEPS: usize = 1200;

fn secs(t: Tick) -> f64 {
    (t.seconds() * 1000.0).round() / 1000.0
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "video.screenActivity",
        label: "Measure Screen Activity",
        menu: &[],
        shortcut: None,
        params: r#"{"clip":id?,"from":seconds?,"to":seconds?,"stepMs":f=100,"region":{"x":px,"y":px,"w":px,"h":px}?,"scale":f=0.25,"threshold":0..255=12,"activeFraction":f=0.0003,"mergeMs":f=300,"samples":bool=true}"#,
        enabled: has_seq,
        run: screen_activity,
        journal: false,
    }]
}

fn grey(rgba: &[u8], w: usize, h: usize, crop: (usize, usize, usize, usize)) -> Vec<u8> {
    let (x0, y0, cw, ch) = crop;
    let mut out = Vec::with_capacity(cw.saturating_mul(ch));
    for y in y0..(y0 + ch).min(h) {
        for x in x0..(x0 + cw).min(w) {
            let i = (y * w + x) * 4;
            let (r, g, b) = (rgba.get(i).copied().unwrap_or(0), rgba.get(i + 1).copied().unwrap_or(0), rgba.get(i + 2).copied().unwrap_or(0));
            out.push(((u32::from(r) * 77 + u32::from(g) * 150 + u32::from(b) * 29) >> 8) as u8);
        }
    }
    out
}

fn screen_activity(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "video.screenActivity";
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let from = time_p(s, p, "from").or_else(|| f64_p(p, "from").map(Tick::from_seconds_f64));
    let to = time_p(s, p, "to").or_else(|| f64_p(p, "to").map(Tick::from_seconds_f64));
    // one clip (`clip`), or the sequence: at each step, the topmost enabled video clip there
    let only = match u64_p(p, "clip").map(ClipId) {
        Some(c) => Some(
            q.find_item(c)
                .filter(|(t, _)| q.track(*t).is_some_and(|t| t.kind == TrackKind::Video))
                .map(|(_, i)| i.clone())
                .ok_or_else(|| bad(cmd, "no such video clip (`clip` is a timeline video clip id)"))?,
        ),
        None => None,
    };
    let (lo, hi) = only.as_ref().map(|c| (c.start, c.end())).unwrap_or((Tick::ZERO, q.duration()));
    let r = TimeRange::from_bounds(from.unwrap_or(lo).max(lo), to.unwrap_or(hi).min(hi));
    if r.duration <= Tick::ZERO {
        return Err(bad(cmd, "the range is empty (or outside the clip)"));
    }
    let step = Tick::from_seconds_f64(f64_p(p, "stepMs").filter(|v| v.is_finite() && *v >= 10.0).unwrap_or(100.0) / 1000.0);
    let n = usize::try_from(r.duration.0 / step.0.max(1)).unwrap_or(usize::MAX).saturating_add(1);
    if n > MAX_STEPS {
        return Err(bad(cmd, format!("{n} steps: at most {MAX_STEPS} per call (shorten the range or raise stepMs)")));
    }
    let scale = f64_p(p, "scale").filter(|v| v.is_finite() && *v > 0.0).unwrap_or(0.25).min(1.0);
    let thr = u64_p(p, "threshold").unwrap_or(12).min(255) as i16;
    let active_at = f64_p(p, "activeFraction").filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(0.0003);
    let merge = Tick::from_seconds_f64(f64_p(p, "mergeMs").filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(300.0) / 1000.0);
    let region = p.get("region").map(|g| {
        let v = |k: &str| g.get(k).and_then(Value::as_f64).filter(|v| v.is_finite()).unwrap_or(0.0).max(0.0);
        (v("x"), v("y"), v("w"), v("h"))
    });
    // (clip, its grey region) of the previous step; a step on another clip is a cut, not activity
    let mut prev: Option<(ClipId, Vec<u8>)> = None;
    // (time, Some((fraction, mean)) or None at a cut / where there is no picture)
    let mut samples: Vec<(Tick, Option<(f64, f64)>)> = Vec::new();
    let mut clips_seen: Vec<u64> = Vec::new();
    for k in 0..n {
        let t = (r.start + Tick(step.0.saturating_mul(k as i64))).min(r.end() - Tick(1));
        let clip = match &only {
            Some(c) => Some(c.clone()),
            None => q.video_tracks.iter().rev().flat_map(|tr| tr.items.iter()).find(|i| i.enabled && i.start <= t && t < i.end()).cloned(),
        };
        let Some((clip, src)) = clip.and_then(|c| s.source(c.item).map(|src| (c, src))) else {
            if k > 0 {
                samples.push((t, None));
            }
            prev = None;
            continue;
        };
        let Some(info) = src.info().video.clone() else { continue };
        let (sw, sh) = (f64::from(info.width.max(1)), f64::from(info.height.max(1)));
        let f = src
            .video_frame(filmcraft_media::FrameRequest { time: clip.source_time_at(t), scale: scale as f32 })
            .map_err(|e| EngineError::Other(format!("{cmd}: can't decode the frame at {:.3} s: {e}", t.seconds())))?;
        let (w, h) = (f.width as usize, f.height as usize);
        let (fx, fy) = (w as f64 / sw, h as f64 / sh);
        let crop = match region {
            Some((x, y, cw, ch)) if cw > 0.0 && ch > 0.0 => {
                ((x * fx) as usize, (y * fy) as usize, ((cw * fx).ceil() as usize).max(1), ((ch * fy).ceil() as usize).max(1))
            }
            _ => (0, 0, w, h),
        };
        let g = grey(&f.to_rgba8(), w, h, crop);
        if !clips_seen.contains(&clip.id.0) {
            clips_seen.push(clip.id.0);
        }
        match &prev {
            Some((pc, pv)) if *pc == clip.id && pv.len() == g.len() && !g.is_empty() => {
                let (mut moved, mut sum) = (0usize, 0u64);
                for (a, b) in pv.iter().zip(&g) {
                    let d = (i16::from(*a) - i16::from(*b)).abs();
                    sum += d as u64;
                    moved += usize::from(d > thr);
                }
                samples.push((t, Some((moved as f64 / g.len() as f64, sum as f64 / g.len() as f64))));
            }
            Some(_) => samples.push((t, None)),
            None => {}
        }
        prev = Some((clip.id, g));
    }
    // active ranges: a sample covers the step that ends at its time
    let mut ranges: Vec<TimeRange> = Vec::new();
    for (t, m) in &samples {
        let Some((frac, _)) = m else { continue };
        if *frac < active_at {
            continue;
        }
        let a = (*t - step).max(r.start);
        match ranges.last_mut() {
            Some(last) if a <= last.end() + merge => *last = TimeRange::from_bounds(last.start, *t),
            _ => ranges.push(TimeRange::from_bounds(a, *t)),
        }
    }
    let active: Tick = ranges.iter().fold(Tick::ZERO, |a, x| a + x.duration);
    let strip: String = samples
        .iter()
        .map(|(_, m)| match m.map(|x| x.0) {
            None => '|',
            Some(x) if x < active_at => '.',
            Some(x) if x < 0.01 => '▂',
            Some(x) if x < 0.1 => '▅',
            _ => '█',
        })
        .collect();
    let want_samples = p.get("samples").and_then(Value::as_bool).unwrap_or(true);
    Ok(json!({
        "clips": clips_seen, "from": secs(r.start), "to": secs(r.end()), "stepMs": (step.seconds() * 1000.0).round(),
        "region": region.map(|(x, y, w, h)| json!({"x": x, "y": y, "w": w, "h": h})),
        "activeSeconds": secs(active),
        "activeRanges": ranges.iter().map(|x| json!({"start": secs(x.start), "end": secs(x.end()), "settlesAt": secs(x.end())})).collect::<Vec<_>>(),
        "strip": strip,
        "columns": ["seconds", "fraction", "mean"],
        "samples": if want_samples {
            json!(samples
                .iter()
                .map(|(t, m)| match m {
                    Some((f, mean)) => json!([secs(*t), (f * 1e5).round() / 1e5, (mean * 100.0).round() / 100.0]),
                    None => json!([secs(*t), null, null]),
                })
                .collect::<Vec<_>>())
        } else { Value::Null },
        "note": "sequence seconds; fraction = share of region pixels that changed by more than `threshold` since the previous step; strip: one char per step ('.' still, ▂ small, ▅ medium, █ large change, '|' a cut: not compared)",
    }))
}
