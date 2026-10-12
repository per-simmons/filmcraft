//! Waveform inspection for agents (`waveform.*`): the numbers behind the timeline's waveforms, so an
//! agent can check its cuts by measurement instead of by looking at screenshots.
//!
//! - `waveform.envelope`: the active sequence's mix over a stretch, as RMS / peak dBFS rows and a
//!   text strip (level, voice, cuts, words), at any resolution down to 5 ms.
//! - `waveform.silences`: every quiet run of the whole mix (dead air), measured like
//!   `qa/deadair.py`: 10 ms RMS windows every 5 ms against a gate 8 dB over the mix's own noise floor
//!   (kept between −50 and −35 dBFS).
//! - `waveform.joins`: every cut between two audio clips, checked against the source's voice map
//!   (`transcript.findPauses` / `transcript.generate` measure it from the waveform) and words: a cut
//!   that clips a word (tail or onset), voice that runs on past a cut, and loose silence kept on
//!   either side beyond Pat's margins.
//! - `waveform.fixJoins`: ripple-trim the joins `waveform.joins` flags onto the voice edges plus the
//!   margins (whole frames, never into speech). One undo step.
//!
//! Levels are dBFS of the mix rendered through the mixer graph exactly as playback and export hear
//! it. Everything is total: a silent, empty or untranscribed sequence gives empty results, never a
//! panic.

use serde_json::{Value, json};

use filmcraft_edit::transcript::{self as tx, SeqWord};
use filmcraft_edit::{Edge, ripple_trim_group};
use filmcraft_project::{ClipId, Sequence, TrackItem, Word};
use filmcraft_time::{FrameRate, Tick, TimeRange};

use crate::commands::{CommandSpec, bad, bool_p, f64_p, has_seq, time_p, u64_p};
use crate::{EngineError, Result, Session};

type Run = fn(&mut Session, &Value) -> Result<Value>;

fn query(id: &'static str, label: &'static str, params: &'static str, run: Run) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: has_seq, run, journal: false }
}

/// Envelope hop: 5 ms.
const HOP_MS: f64 = 5.0;
/// Longest stretch `waveform.envelope` renders at once.
const MAX_ENVELOPE_SECONDS: f64 = 600.0;
/// Most rows `waveform.envelope` returns.
const MAX_POINTS: usize = 4000;
/// Below this a frame is digital silence (a closed noise gate), left out of the floor.
const LIVE_DB: f64 = -90.0;
/// A word or voice edge this close to a cut is on it, not across it.
const EDGE_TOLERANCE_MS: f64 = 10.0;

fn ms(t: Tick) -> f64 {
    (t.seconds() * 10_000.0).round() / 10.0
}

fn secs(t: Tick) -> f64 {
    (t.seconds() * 10_000.0).round() / 10_000.0
}

fn db(x: f64) -> f64 {
    let v = if x > 1e-10 { 20.0 * x.log10() } else { -200.0 };
    (v.max(-120.0) * 10.0).round() / 10.0
}

fn tc(s: &Session, t: Tick) -> String {
    let rate = s.sequence_rate();
    let df = s.active_sequence().map(|q| q.settings.drop_frame).unwrap_or(false);
    filmcraft_time::format_timecode_frames(rate.frame_at(t), rate, df)
}

/// Energy of one 5 ms block of the mix: sum of squares over every channel and sample, the sample
/// count (channels × samples) and the peak.
#[derive(Clone, Copy, Debug, Default)]
struct Block {
    sq: f64,
    n: usize,
    peak: f32,
}

impl Block {
    fn add(&mut self, o: &Block) {
        self.sq += o.sq;
        self.n += o.n;
        self.peak = self.peak.max(o.peak);
    }
    fn rms(&self) -> f64 {
        if self.n == 0 { 0.0 } else { (self.sq / self.n as f64).sqrt() }
    }
}

/// Render `q`'s stereo mix over `[from, to)` a second at a time, handing each buffer to `f`.
/// Returns the first sample (sequence rate).
fn render_mix(s: &Session, q: &Sequence, from: Tick, to: Tick, mut f: impl FnMut(&filmcraft_frame::AudioBuffer)) -> i64 {
    let sr = i64::from(q.settings.sample_rate.max(1));
    let start = from.max(Tick::ZERO).to_units_floor(sr);
    let end = to.to_units_floor(sr).max(start);
    let provider = s.media.full_res_provider(s.project.clone(), s.services.clone());
    let mut pos = start;
    while pos < end {
        let n = (end - pos).min(sr);
        let Ok(frames) = usize::try_from(n) else { break };
        f(&filmcraft_render::audio::mix_sequence(&s.project, q, pos, frames, &provider));
        pos += n;
    }
    start
}

/// The active sequence's mix over `[from, to)` as 5 ms blocks.
fn blocks(s: &Session, from: Tick, to: Tick) -> Result<(Vec<Block>, Tick)> {
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let sr = i64::from(q.settings.sample_rate.max(1));
    let hop = (sr / 200).max(1);
    let span = to.to_units_floor(sr) - from.max(Tick::ZERO).to_units_floor(sr);
    let cap = usize::try_from(span / hop + 1).unwrap_or(0).min(MAX_ENVELOPE_SECONDS as usize * 200 * 12);
    let mut out: Vec<Block> = Vec::with_capacity(cap);
    let mut cur = Block::default();
    let mut in_block = 0i64;
    let start = render_mix(s, q, from, to, |b| {
        let frames = b.channels.first().map(Vec::len).unwrap_or(0);
        for i in 0..frames {
            for ch in &b.channels {
                let v = ch.get(i).copied().unwrap_or(0.0);
                let v = if v.is_finite() { v } else { 0.0 };
                cur.sq += f64::from(v) * f64::from(v);
                cur.n += 1;
                cur.peak = cur.peak.max(v.abs());
            }
            in_block += 1;
            if in_block == hop {
                out.push(cur);
                cur = Block::default();
                in_block = 0;
            }
        }
    });
    if in_block > 0 {
        out.push(cur);
    }
    Ok((out, Tick::from_units(start, sr)))
}

/// 10 ms window levels (dBFS), one per 5 ms hop: window `i` covers blocks `i` and `i + 1`.
fn window_db(b: &[Block]) -> Vec<f64> {
    (0..b.len())
        .map(|i| {
            let mut w = b.get(i).copied().unwrap_or_default();
            if let Some(n) = b.get(i + 1) {
                w.add(n);
            }
            db(w.rms())
        })
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted.get(i).copied()
}

/// (noise floor, speech level) of window levels: their 10th and 90th percentiles, digital silence
/// left out.
fn levels(win: &[f64]) -> (f64, f64) {
    let mut live: Vec<f64> = win.iter().copied().filter(|v| *v > LIVE_DB).collect();
    live.sort_by(f64::total_cmp);
    (percentile(&live, 0.1).unwrap_or(-60.0), percentile(&live, 0.9).unwrap_or(-20.0))
}

/// Floor and speech level with the floor taken where the voice map has no voice (when there are at
/// least 50 ms of such windows). An edited sequence has had its silence cut out: the 10th
/// percentile of all its windows is then the quiet end of speech, which would put the gate on top
/// of word tails. Returns (floor, speech, measured outside the voice).
fn levels_outside_voice(win: &[f64], t0: Tick, voice: &[TimeRange]) -> (f64, f64, bool) {
    let (floor_all, speech) = levels(win);
    let hop = Tick::from_seconds_f64(HOP_MS / 1000.0);
    let mut quiet: Vec<f64> = win
        .iter()
        .enumerate()
        .filter(|(i, v)| {
            let c = t0 + Tick(hop.0.saturating_mul(*i as i64)) + hop;
            **v > LIVE_DB && !voice.iter().any(|r| r.contains(c))
        })
        .map(|(_, v)| *v)
        .collect();
    if voice.is_empty() || quiet.len() < 10 {
        return (floor_all, speech, false);
    }
    quiet.sort_by(f64::total_cmp);
    (percentile(&quiet, 0.1).unwrap_or(floor_all).min(floor_all), speech, true)
}

/// The deadair.py gate: 8 dB over the floor, kept between −50 and −35 dBFS.
fn auto_gate(floor: f64) -> f64 {
    (floor + 8.0).clamp(-50.0, -35.0)
}

/// `from`/`to` (or `around` ± `span`/2) in sequence time, defaulting to the whole sequence.
fn range_p(s: &Session, p: &Value, cmd: &str, default_span: f64) -> Result<TimeRange> {
    let dur = s.active_sequence().map(|q| q.duration()).unwrap_or_default();
    let (a, b) = if let Some(c) = time_p(s, p, "around").or_else(|| f64_p(p, "around").map(Tick::from_seconds_f64)) {
        let half = Tick::from_seconds_f64(f64_p(p, "span").filter(|v| v.is_finite() && *v > 0.0).unwrap_or(default_span) / 2.0);
        (c - half, c + half)
    } else {
        let a = time_p(s, p, "from").or_else(|| f64_p(p, "from").map(Tick::from_seconds_f64)).unwrap_or(Tick::ZERO);
        let b = time_p(s, p, "to").or_else(|| f64_p(p, "to").map(Tick::from_seconds_f64)).unwrap_or(dur);
        (a, b)
    };
    let (a, b) = (a.max(Tick::ZERO), b.min(dur));
    if b <= a {
        return Err(bad(cmd, "the range is empty (`from` < `to`, inside the sequence)"));
    }
    Ok(TimeRange::from_bounds(a, b))
}

/// Where cuts are: the starts and ends of enabled audio clips, sorted and deduplicated.
fn edit_points(q: &Sequence) -> Vec<Tick> {
    let mut v: Vec<Tick> = q.audio_tracks.iter().flat_map(|t| t.items.iter()).filter(|i| i.enabled).flat_map(|i| [i.start, i.end()]).collect();
    v.sort();
    v.dedup();
    v
}

fn word_at(words: &[SeqWord], t: Tick) -> Option<&SeqWord> {
    words.iter().find(|w| w.start <= t && t < w.end)
}

fn word_before(words: &[SeqWord], t: Tick) -> Option<&SeqWord> {
    words.iter().rev().find(|w| w.end <= t + Tick::from_seconds_f64(EDGE_TOLERANCE_MS / 1000.0))
}

fn word_after(words: &[SeqWord], t: Tick) -> Option<&SeqWord> {
    words.iter().find(|w| w.start + Tick::from_seconds_f64(EDGE_TOLERANCE_MS / 1000.0) >= t)
}

// ---------------------------------------------------------------------------------------------
// waveform.envelope
// ---------------------------------------------------------------------------------------------

const BARS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

fn envelope(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "waveform.envelope";
    let r = range_p(s, p, cmd, 2.0)?;
    if r.duration.seconds() > MAX_ENVELOPE_SECONDS {
        return Err(bad(cmd, format!("at most {MAX_ENVELOPE_SECONDS} s at a time (use `waveform.silences` for the whole sequence)")));
    }
    let (blk, t0) = blocks(s, r.start, r.end())?;
    let hop = Tick::from_seconds_f64(HOP_MS / 1000.0);
    // bucket size: `windowMs`, else enough to give `points` rows (default 120)
    let points = u64_p(p, "points").map(|v| v as usize).unwrap_or(120).clamp(1, MAX_POINTS);
    let per = match f64_p(p, "windowMs").filter(|v| v.is_finite() && *v > 0.0) {
        Some(w) => ((w / HOP_MS).round() as usize).max(1),
        None => blk.len().div_ceil(points).max(1),
    };
    if blk.len().div_ceil(per) > MAX_POINTS {
        return Err(bad(cmd, format!("that is more than {MAX_POINTS} rows: widen `windowMs` or shorten the range")));
    }
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let words = tx::sequence_words(q, &s.project.transcripts);
    let voice = tx::sequence_voice(q, &s.project.transcripts);
    let cuts: Vec<Tick> = edit_points(q).into_iter().filter(|t| r.contains(*t) || *t == r.end()).collect();
    let (floor, speech, _) = levels_outside_voice(&window_db(&blk), t0, &voice.spans);
    let mut rows = Vec::new();
    let (mut lvl, mut vline, mut cline) = (String::new(), String::new(), String::new());
    for (i, chunk) in blk.chunks(per).enumerate() {
        let mut b = Block::default();
        chunk.iter().for_each(|c| b.add(c));
        let a = t0 + Tick(hop.0.saturating_mul((i * per) as i64));
        let e = a + Tick(hop.0.saturating_mul(chunk.len() as i64));
        let bucket = TimeRange::from_bounds(a, e);
        let voiced = voice.spans.iter().any(|v| v.overlaps(&bucket));
        let cut = cuts.iter().any(|c| bucket.contains(*c));
        let (rms, peak) = (db(b.rms()), db(f64::from(b.peak)));
        rows.push(json!([secs(a), rms, peak, u8::from(voiced), u8::from(cut)]));
        let lv = ((rms - floor) / ((speech + 6.0) - floor).max(1.0) * 8.0).round().clamp(0.0, 8.0) as usize;
        lvl.push(BARS.get(lv).copied().unwrap_or(' '));
        vline.push(if voiced { '=' } else { '.' });
        cline.push(if cut { '|' } else { ' ' });
    }
    let in_range: Vec<Value> =
        words.iter().filter(|w| w.end > r.start && w.start < r.end()).map(|w| json!({"text": w.text, "start": secs(w.start), "end": secs(w.end)})).collect();
    let voiced: Vec<Value> = voice.spans.iter().filter_map(|v| v.intersect(&r)).map(|v| json!([secs(v.start), secs(v.end())])).collect();
    Ok(json!({
        "from": secs(r.start), "to": secs(r.end()),
        "fromTimecode": tc(s, r.start), "toTimecode": tc(s, r.end()),
        "windowMs": per as f64 * HOP_MS,
        "floorDb": floor, "speechDb": speech, "gateDb": auto_gate(floor),
        "columns": ["seconds", "rmsDb", "peakDb", "voice", "cut"],
        "rows": rows,
        "strip": {"level": lvl, "voice": vline, "cuts": cline},
        "voice": voiced,
        "cuts": cuts.iter().map(|t| secs(*t)).collect::<Vec<_>>(),
        "words": in_range,
        "note": "levels are dBFS of the mix as playback hears it; strip: one char per row (level ▁…█ from the floor to the speech level, voice '=' where the waveform voice map has voice, cuts '|')",
    }))
}

// ---------------------------------------------------------------------------------------------
// waveform.silences
// ---------------------------------------------------------------------------------------------

fn silences(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "waveform.silences";
    let r = range_p(s, p, cmd, 2.0)?;
    let min = Tick::from_seconds_f64(f64_p(p, "minSeconds").filter(|v| v.is_finite() && *v > 0.0).unwrap_or(0.15));
    let limit = u64_p(p, "maxResults").map(|v| v as usize).unwrap_or(200).clamp(1, 5000);
    let (blk, t0) = blocks(s, r.start, r.end())?;
    let win = window_db(&blk);
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let voice = tx::sequence_voice(q, &s.project.transcripts);
    let (floor, speech, outside) = levels_outside_voice(&win, t0, &voice.spans);
    let gate = f64_p(p, "gateDb").filter(|v| v.is_finite()).unwrap_or_else(|| auto_gate(floor));
    let hop = Tick::from_seconds_f64(HOP_MS / 1000.0);
    let at = |i: usize| t0 + Tick(hop.0.saturating_mul(i as i64));
    let words = tx::sequence_words(q, &s.project.transcripts);
    let cuts = edit_points(q);
    let mut runs: Vec<TimeRange> = Vec::new();
    let mut start: Option<usize> = None;
    for i in 0..=win.len() {
        let quiet = win.get(i).is_some_and(|v| *v < gate);
        match (quiet, start) {
            (true, None) => start = Some(i),
            (false, Some(a)) => {
                // window i-1 is the last quiet one; it reaches one hop past its own start
                let rr = TimeRange::from_bounds(at(a), at(i + 1).min(r.end()));
                if rr.duration >= min {
                    runs.push(rr);
                }
                start = None;
            }
            _ => {}
        }
    }
    let total: Tick = runs.iter().fold(Tick::ZERO, |a, r| a + r.duration);
    let longest = runs.iter().map(|r| r.duration).max().unwrap_or_default();
    let dur = s.active_sequence().map(|q| q.duration()).unwrap_or_default();
    let list: Vec<Value> = runs
        .iter()
        .take(limit)
        .map(|rr| {
            let kind = if rr.start <= Tick::from_seconds_f64(0.01) {
                "head"
            } else if rr.end() + Tick::from_seconds_f64(0.01) >= dur {
                "tail"
            } else if cuts.iter().any(|c| rr.contains(*c)) {
                "acrossCut"
            } else {
                "inClip"
            };
            json!({
                "start": secs(rr.start), "end": secs(rr.end()), "ms": ms(rr.duration),
                "timecode": tc(s, rr.start), "kind": kind,
                "after": word_before(&words, rr.start).map(|w| w.text.clone()),
                "before": word_after(&words, rr.end()).map(|w| w.text.clone()),
            })
        })
        .collect();
    Ok(json!({
        "from": secs(r.start), "to": secs(r.end()),
        "minMs": ms(min), "gateDb": gate, "floorDb": floor, "speechDb": speech,
        "floorFrom": if outside { "outsideVoice" } else { "allAudio" },
        "count": runs.len(), "totalSeconds": secs(total), "longestMs": ms(longest),
        "durationSeconds": secs(r.duration),
        "runs": list,
        "truncated": runs.len() > limit,
    }))
}

// ---------------------------------------------------------------------------------------------
// waveform.joins / waveform.fixJoins
// ---------------------------------------------------------------------------------------------

/// Pat's margins (`keepAfterSeconds`, `keepBeforeSeconds`; else Settings ▸ pause keep after/before).
fn margins(s: &Session, p: &Value) -> (Tick, Tick) {
    let ma = &s.prefs.media_analysis;
    let sec = |k: &str| f64_p(p, k).filter(|v| v.is_finite() && *v >= 0.0).map(Tick::from_seconds_f64);
    let msv = |v: u32| Tick::from_seconds_f64(f64::from(v) / 1000.0);
    (sec("keepAfterSeconds").unwrap_or(msv(ma.pause_keep_after_ms)), sec("keepBeforeSeconds").unwrap_or(msv(ma.pause_keep_before_ms)))
}

/// One clip's source voice map and words (media time), when it can be read: transcribed, forwards,
/// 100 % speed, no frame hold.
struct SourceVoice<'a> {
    voice: Vec<(Tick, Tick)>,
    words: &'a [Word],
}

fn source_voice<'a>(s: &'a Session, it: &TrackItem) -> Option<SourceVoice<'a>> {
    if it.reverse || it.frame_hold.is_some() || (it.speed - 1.0).abs() > 1e-9 {
        return None;
    }
    let root = s.project.resolve_media(it.item).map(|(r, _, _)| r).unwrap_or(it.item);
    let t = s.project.transcripts.get(&root).or_else(|| s.project.transcripts.get(&it.item))?;
    let voice = if t.voice.is_empty() { t.words.iter().map(|w| (w.start, w.end.max(w.start))).collect() } else { t.voice.clone() };
    Some(SourceVoice { voice, words: &t.words })
}

/// What one side of a cut looks like in the source.
#[derive(Debug, Default, Clone)]
struct Side {
    /// A word the cut goes through: (text, its edge across the cut, media time).
    cut_word: Option<(String, Tick)>,
    /// Voice running on across the cut (media time it ends / starts).
    runs_on: Option<Tick>,
    /// Quiet between the cut and the nearest voice inside the clip.
    silence: Option<Tick>,
    /// Nearest voice edge inside the clip (media time).
    voice_edge: Option<Tick>,
    /// The next (tail) / previous (onset) word's edge beyond the cut, which a repair must not reach.
    limit: Option<Tick>,
}

/// The tail of a clip ending at media time `out` (clip media from `inn`).
fn tail_side(v: &SourceVoice, inn: Tick, out: Tick) -> Side {
    let tol = Tick::from_seconds_f64(EDGE_TOLERANCE_MS / 1000.0);
    let mut sd = Side::default();
    if let Some(w) = v.words.iter().find(|w| w.start + tol < out && w.end > out + tol) {
        sd.cut_word = Some((w.text.clone(), w.end));
    }
    sd.limit = v.words.iter().find(|w| w.start >= out && sd.cut_word.as_ref().is_none_or(|(_, e)| w.start >= *e)).map(|w| w.start);
    if let Some((_, b)) = v.voice.iter().find(|(a, b)| *a < out && *b > out + tol) {
        sd.runs_on = Some(*b);
    } else if let Some((_, b)) = v.voice.iter().rev().find(|(a, b)| *b <= out + tol && *b > inn && *a < out) {
        sd.voice_edge = Some(*b);
        sd.silence = Some((out - *b).max(Tick::ZERO));
    } else {
        sd.silence = Some(out - inn);
    }
    sd
}

/// The head of a clip starting at media time `inn` (clip media to `out`).
fn head_side(v: &SourceVoice, inn: Tick, out: Tick) -> Side {
    let tol = Tick::from_seconds_f64(EDGE_TOLERANCE_MS / 1000.0);
    let mut sd = Side::default();
    if let Some(w) = v.words.iter().find(|w| w.start + tol < inn && w.end > inn + tol) {
        sd.cut_word = Some((w.text.clone(), w.start));
    }
    sd.limit = v.words.iter().rev().find(|w| w.end <= inn && sd.cut_word.as_ref().is_none_or(|(_, st)| w.end <= *st)).map(|w| w.end);
    if let Some((a, _)) = v.voice.iter().find(|(a, b)| *a + tol < inn && *b > inn) {
        sd.runs_on = Some(*a);
    } else if let Some((a, _)) = v.voice.iter().find(|(a, b)| *a + tol >= inn && *a < out && *b > inn) {
        sd.voice_edge = Some(*a);
        sd.silence = Some((*a - inn).max(Tick::ZERO));
    } else {
        sd.silence = Some(out - inn);
    }
    sd
}

/// A planned change to one edge: positive = longer (Out later / In earlier), in whole frames.
fn frames_toward_zero(d: Tick, frame: Tick) -> Tick {
    if frame <= Tick::ZERO {
        return Tick::ZERO;
    }
    Tick((d.0 / frame.0).saturating_mul(frame.0))
}

/// Extend by at least `need` (whole frames) without passing `room`.
fn frames_extend(need: Tick, room: Option<Tick>, frame: Tick) -> Tick {
    if frame <= Tick::ZERO || need <= Tick::ZERO {
        return Tick::ZERO;
    }
    let up = Tick((need.0 / frame.0 + i64::from(need.0 % frame.0 != 0)).saturating_mul(frame.0));
    match room {
        Some(r) if up > r => frames_toward_zero(r, frame),
        _ => up,
    }
}

struct Join {
    track: usize,
    left: Option<ClipId>,
    right: Option<ClipId>,
    at: Tick,
    /// Timeline gap between the clips (0 when they touch).
    gap: Tick,
    tail: Option<Side>,
    head: Option<Side>,
    /// Planned Out-edge change of the left clip and In-edge change of the right (positive = longer).
    tail_fix: Tick,
    head_fix: Tick,
    issues: Vec<&'static str>,
}

fn analyse_joins(s: &Session, p: &Value) -> Result<Vec<Join>> {
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let rate: FrameRate = s.sequence_rate();
    let frame = rate.frame_duration();
    let (keep_after, keep_before) = margins(s, p);
    let slack = frame; // loose = more than a frame over the margin
    let only = u64_p(p, "track").map(|v| v as usize);
    let tightening = bool_p(p, "tighten").unwrap_or(true);
    let mut out = Vec::new();
    for (ti, t) in q.audio_tracks.iter().enumerate() {
        if only.is_some_and(|o| o != ti) || t.locked {
            continue;
        }
        let mut items: Vec<&TrackItem> = t.items.iter().filter(|i| i.enabled).collect();
        items.sort_by_key(|i| i.start);
        let mut prev: Option<&TrackItem> = None;
        for it in items.iter().copied().map(Some).chain(std::iter::once(None)) {
            let (l, r) = (prev, it);
            prev = it;
            if l.is_none() && r.is_none() {
                continue;
            }
            // a through edit (same media, contiguous) is not a cut
            if let (Some(a), Some(b)) = (l, r)
                && a.item == b.item
                && a.end() == b.start
                && a.source_out() == b.source_in
                && (a.speed - b.speed).abs() < 1e-9
            {
                continue;
            }
            let at = r.map(|b| b.start).or(l.map(|a| a.end())).unwrap_or_default();
            let gap = match (l, r) {
                (Some(a), Some(b)) => (b.start - a.end()).max(Tick::ZERO),
                _ => Tick::ZERO,
            };
            let tail = l.and_then(|a| source_voice(s, a).map(|v| tail_side(&v, a.source_in, a.source_out())));
            let head = r.and_then(|b| source_voice(s, b).map(|v| head_side(&v, b.source_in, b.source_out())));
            let mut j = Join {
                track: ti,
                left: l.map(|a| a.id),
                right: r.map(|b| b.id),
                at,
                gap,
                tail,
                head,
                tail_fix: Tick::ZERO,
                head_fix: Tick::ZERO,
                issues: vec![],
            };
            if let (Some(sd), Some(a)) = (&j.tail, l) {
                let out_t = a.source_out();
                if let Some((_, end)) = &sd.cut_word {
                    j.issues.push("clipsWordTail");
                    let room = sd.limit.map(|lim| (lim - out_t).max(Tick::ZERO));
                    j.tail_fix = frames_extend(*end + keep_after - out_t, room, frame);
                } else if sd.runs_on.is_some() {
                    j.issues.push("voiceRunsOnTail");
                } else if let Some(sil) = sd.silence
                    && sil > keep_after + slack
                    && sd.voice_edge.is_some()
                {
                    j.issues.push("looseTail");
                    if tightening {
                        j.tail_fix = -frames_toward_zero(sil - keep_after, frame);
                    }
                }
            }
            if let (Some(sd), Some(b)) = (&j.head, r) {
                let in_t = b.source_in;
                if let Some((_, st)) = &sd.cut_word {
                    j.issues.push("clipsWordOnset");
                    let room = sd.limit.map(|lim| (in_t - lim).max(Tick::ZERO));
                    j.head_fix = frames_extend(in_t - (*st - keep_before), room, frame);
                } else if sd.runs_on.is_some() {
                    j.issues.push("voiceRunsOnOnset");
                } else if let Some(sil) = sd.silence
                    && sil > keep_before + slack
                    && sd.voice_edge.is_some()
                {
                    j.issues.push("looseHead");
                    if tightening {
                        j.head_fix = -frames_toward_zero(sil - keep_before, frame);
                    }
                }
            }
            if gap > Tick::ZERO {
                j.issues.push("timelineGap");
            }
            if l.is_some_and(|a| source_voice(s, a).is_none()) || r.is_some_and(|b| source_voice(s, b).is_none()) {
                j.issues.push("noVoiceMap");
            }
            out.push(j);
        }
    }
    out.sort_by_key(|j| (j.at, j.track));
    Ok(out)
}

fn side_json(sd: &Option<Side>, edge: Tick, after_cut: bool) -> Value {
    let Some(sd) = sd else { return Value::Null };
    json!({
        "clipsWord": sd.cut_word.as_ref().map(|(w, e)| json!({"text": w, "lostMs": ms(if after_cut { edge - *e } else { *e - edge })})),
        "voiceRunsOnMs": sd.runs_on.map(|t| ms(if after_cut { edge - t } else { t - edge })),
        "silenceMs": sd.silence.map(ms),
    })
}

fn levels_at(s: &Session, at: Tick) -> (f64, f64) {
    let w = Tick::from_seconds_f64(0.015);
    let lv = |a: Tick, b: Tick| -> f64 {
        blocks(s, a, b)
            .map(|(bl, _)| {
                let mut x = Block::default();
                bl.iter().for_each(|c| x.add(c));
                db(x.rms())
            })
            .unwrap_or(-120.0)
    };
    (lv((at - w).max(Tick::ZERO), at), lv(at, at + w))
}

fn joins(s: &mut Session, p: &Value) -> Result<Value> {
    let js = analyse_joins(s, p)?;
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let words = tx::sequence_words(q, &s.project.transcripts);
    let with_levels = bool_p(p, "levels").unwrap_or(true);
    let all = bool_p(p, "all").unwrap_or(false);
    let (keep_after, keep_before) = margins(s, p);
    let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut list = Vec::new();
    for (i, j) in js.iter().enumerate() {
        for k in &j.issues {
            *counts.entry(k).or_default() += 1;
        }
        let actionable = j.issues.iter().any(|k| !matches!(*k, "noVoiceMap"));
        if !all && !actionable {
            continue;
        }
        let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
        let src_out = j.left.and_then(|c| q.find_item(c)).map(|(_, a)| a.source_out()).unwrap_or_default();
        let src_in = j.right.and_then(|c| q.find_item(c)).map(|(_, b)| b.source_in).unwrap_or_default();
        let (lb, la) = if with_levels { levels_at(s, j.at) } else { (f64::NAN, f64::NAN) };
        let gap_ms = j.tail.as_ref().and_then(|t| t.silence).unwrap_or_default() + j.head.as_ref().and_then(|h| h.silence).unwrap_or_default() + j.gap;
        list.push(json!({
            "join": i, "track": j.track, "seconds": secs(j.at), "timecode": tc(s, j.at),
            "left": j.left.map(|c| c.0), "right": j.right.map(|c| c.0),
            "wordBefore": word_before(&words, j.at).map(|w| w.text.clone()),
            "wordAfter": word_after(&words, j.at).map(|w| w.text.clone()),
            "wordAcross": word_at(&words, j.at).map(|w| w.text.clone()),
            "tail": side_json(&j.tail, src_out, false),
            "head": side_json(&j.head, src_in, true),
            "quietAcrossMs": ms(gap_ms),
            "timelineGapMs": ms(j.gap),
            "levelBeforeDb": if lb.is_finite() { json!(lb) } else { Value::Null },
            "levelAfterDb": if la.is_finite() { json!(la) } else { Value::Null },
            "issues": j.issues,
            "fix": {"tailMs": ms(j.tail_fix), "headMs": ms(j.head_fix)},
        }));
    }
    Ok(json!({
        "joins": js.len(),
        "issues": counts,
        "keepAfterMs": ms(keep_after), "keepBeforeMs": ms(keep_before),
        "list": list,
        "note": "tail = end of the clip before the cut, head = start of the clip after it. clipsWord: the cut goes through a word (lostMs of it is cut off). voiceRunsOn: the waveform's voice continues across the cut with no word through it (usually the removed retake's first sound; check with waveform.envelope). fix: what waveform.fixJoins would do (+ = longer, whole frames).",
    }))
}

fn fix_joins(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "waveform.fixJoins";
    let js = analyse_joins(s, p)?;
    let pick: Option<Vec<usize>> = p.get("joins").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect());
    let clipped_only = bool_p(p, "clippedOnly").unwrap_or(false);
    let dry = bool_p(p, "dryRun").unwrap_or(false);
    // the plan, right to left so a ripple never moves a join still to do
    let mut plan: Vec<(usize, Option<ClipId>, Tick, Option<ClipId>, Tick)> = Vec::new();
    for (i, j) in js.iter().enumerate() {
        if pick.as_ref().is_some_and(|v| !v.contains(&i)) {
            continue;
        }
        let keep = |d: Tick| if clipped_only && d < Tick::ZERO { Tick::ZERO } else { d };
        let (tf, hf) = (keep(j.tail_fix), keep(j.head_fix));
        if tf != Tick::ZERO || hf != Tick::ZERO {
            plan.push((i, j.left, tf, j.right, hf));
        }
    }
    plan.sort_by_key(|x| std::cmp::Reverse(js.get(x.0).map(|j| j.at).unwrap_or_default()));
    let summary: Vec<Value> = plan.iter().map(|(i, _, tf, _, hf)| json!({"join": i, "tailMs": ms(*tf), "headMs": ms(*hf)})).collect();
    if dry || plan.is_empty() {
        return Ok(json!({"dryRun": dry, "planned": summary, "applied": 0}));
    }
    let before = s.active_sequence().map(|q| q.duration()).unwrap_or_default();
    let group = |q: &Sequence, c: ClipId| -> Vec<ClipId> {
        let link = q.find_item(c).and_then(|(_, i)| i.link);
        let mut v = vec![c];
        if let Some(l) = link {
            for t in q.all_tracks() {
                for i in &t.items {
                    if i.link == Some(l) && i.id != c {
                        v.push(i.id);
                    }
                }
            }
        }
        v
    };
    let applied = s.edit_sequence("Fix Joins", move |q, ctx, _| {
        let mut applied = Vec::new();
        for (i, l, tf, r, hf) in plan {
            // the right clip's head first: it is later in time, so the left clip's tail stays put
            let mut done = (Tick::ZERO, Tick::ZERO);
            if let Some(b) = r
                && hf != Tick::ZERO
            {
                let g = group(q, b);
                done.1 = -ripple_trim_group(q, &g, Edge::In, -hf, ctx).map_err(|e| EngineError::Other(e.to_string()))?;
            }
            if let Some(a) = l
                && tf != Tick::ZERO
            {
                let g = group(q, a);
                done.0 = ripple_trim_group(q, &g, Edge::Out, tf, ctx).map_err(|e| EngineError::Other(e.to_string()))?;
            }
            applied.push(json!({"join": i, "tailMs": ms(done.0), "headMs": ms(done.1)}));
        }
        Ok(applied)
    })?;
    let after = s.active_sequence().map(|q| q.duration()).unwrap_or_default();
    if applied.is_empty() {
        return Err(bad(cmd, "nothing could be trimmed"));
    }
    Ok(json!({
        "applied": applied.len(), "joins": applied,
        "durationBefore": secs(before), "durationAfter": secs(after),
        "note": "re-run waveform.joins to confirm; Undo reverts all of it",
    }))
}

// ---------------------------------------------------------------------------------------------
// waveform.loudness
// ---------------------------------------------------------------------------------------------

/// EBU R128 / BS.1770 loudness of `q`'s mix over `r` (true peak 4× oversampled).
fn measure_mix(s: &Session, q: &Sequence, r: TimeRange) -> filmcraft_audio_dsp::loudness::LoudnessSummary {
    let sr = f64::from(q.settings.sample_rate.max(1));
    let mut m = filmcraft_audio_dsp::LoudnessMeter::new(sr, 2);
    render_mix(s, q, r.start, r.end(), |b| {
        let ch: Vec<&[f32]> = b.channels.iter().map(Vec::as_slice).collect();
        m.process(&ch);
    });
    m.summary()
}

fn finite(v: f64) -> Value {
    if v.is_finite() { json!((v * 100.0).round() / 100.0) } else { Value::Null }
}

fn summary_json(l: &filmcraft_audio_dsp::loudness::LoudnessSummary, target: Option<(f64, f64)>) -> Value {
    let mut v = json!({
        "integratedLufs": finite(l.integrated_lufs),
        "loudnessRangeLu": finite(l.loudness_range_lu),
        "maxMomentaryLufs": finite(l.max_momentary_lufs),
        "maxShortTermLufs": finite(l.max_short_term_lufs),
        "samplePeakDbfs": finite(l.sample_peak_dbfs),
        "truePeakDbtp": finite(l.true_peak_dbtp),
    });
    if let Some((lufs, tp)) = target
        && l.integrated_lufs.is_finite()
    {
        let gain = lufs - l.integrated_lufs;
        let peak_after = l.true_peak_dbtp + gain;
        v["target"] = json!({
            "lufs": lufs, "truePeakDbtp": tp,
            "gainDb": finite(gain),
            "truePeakAfterGainDbtp": finite(peak_after),
            "needsLimiter": peak_after > tp,
            "meets": gain.abs() <= 0.5 && l.true_peak_dbtp <= tp,
        });
    }
    v
}

fn loudness(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "waveform.loudness";
    let r = range_p(s, p, cmd, 2.0)?;
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?.clone();
    let target = match (f64_p(p, "targetLufs"), f64_p(p, "targetTruePeakDbtp")) {
        (None, None) => Some((-14.0, -1.5)),
        (l, t) => Some((l.unwrap_or(-14.0), t.unwrap_or(-1.5))),
    };
    // `tracks`: measure only these audio tracks (0 = A1), as if the others were muted
    let solo = |keep: &[usize]| {
        let mut q2 = q.clone();
        for (i, t) in q2.audio_tracks.iter_mut().enumerate() {
            if !keep.contains(&i) {
                t.items.iter_mut().for_each(|it| it.enabled = false);
            }
        }
        q2
    };
    let tracks: Option<Vec<usize>> = p.get("tracks").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect());
    let main = match &tracks {
        Some(t) => measure_mix(s, &solo(t), r),
        None => measure_mix(s, &q, r),
    };
    let mut out = summary_json(&main, target);
    out["from"] = json!(secs(r.start));
    out["to"] = json!(secs(r.end()));
    out["measured"] = match &tracks {
        Some(t) => json!(t.iter().map(|i| format!("A{}", i + 1)).collect::<Vec<_>>()),
        None => json!("mix"),
    };
    if bool_p(p, "perTrack").unwrap_or(false) {
        let per: Vec<Value> = (0..q.audio_tracks.len())
            .filter(|i| q.audio_tracks.get(*i).is_some_and(|t| t.items.iter().any(|it| it.enabled)))
            .map(|i| {
                let mut v = summary_json(&measure_mix(s, &solo(&[i]), r), None);
                v["track"] = json!(format!("A{}", i + 1));
                v["name"] = json!(q.audio_tracks.get(i).map(|t| t.name.clone()));
                v
            })
            .collect();
        out["perTrack"] = json!(per);
    }
    out["note"] = json!(
        "BS.1770-4 / EBU R128 of the mix through the mixer graph (faders, effects, Mix fader) as export hears it; target defaults to Pat's final audio: -14 LUFS, -1.5 dBTP"
    );
    Ok(out)
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        query(
            "waveform.envelope",
            "Waveform Envelope",
            r#"{"from":seconds?,"to":seconds?,"around":seconds?,"span":seconds=2,"points":int=120,"windowMs":f?}"#,
            envelope,
        ),
        query(
            "waveform.loudness",
            "Measure Loudness",
            r#"{"from":seconds?,"to":seconds?,"tracks":[int]?,"perTrack":bool?,"targetLufs":f=-14,"targetTruePeakDbtp":f=-1.5}"#,
            loudness,
        ),
        query("waveform.silences", "Find Dead Air", r#"{"minSeconds":f=0.15,"gateDb":f?,"from":seconds?,"to":seconds?,"maxResults":int=200}"#, silences),
        query(
            "waveform.joins",
            "Audit Cuts Against the Waveform",
            r#"{"track":int?,"all":bool?,"levels":bool=true,"keepAfterSeconds":f?,"keepBeforeSeconds":f?,"tighten":bool=true}"#,
            joins,
        ),
        CommandSpec {
            id: "waveform.fixJoins",
            label: "Fix Cuts to the Waveform",
            menu: &[],
            shortcut: None,
            params: r#"{"joins":[int]?,"dryRun":bool?,"clippedOnly":bool?,"tighten":bool=true,"keepAfterSeconds":f?,"keepBeforeSeconds":f?,"track":int?}"#,
            enabled: has_seq,
            run: fix_joins,
            journal: true,
        },
    ]
}
