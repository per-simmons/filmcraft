//! Staggered text animation: a text layer's words, characters or lines enter (and optionally
//! leave) one after another, each with its own eased rise, fade and scale.
//!
//! The animation is stored on the layer ([`LayerExtra::anim`](crate::graphic_design::LayerExtra))
//! instead of keyframes, so it needs no per-word layers and follows the text when it is edited:
//! the renderer splits the laid-out text into units at draw time. Timing is exact [`Tick`] time
//! relative to the graphic clip: the entrance counts from the clip start, the exit ends at the
//! clip end, so trimming or extending the clip keeps both intact.
//!
//! [`unit_state`] is the pure timing model: given a unit's index among `n` units and the clip
//! time, it returns the unit's offset, opacity and scale. Renderers apply that to the unit's
//! glyphs; queries (`graphics.textLayout`) use it to report animated word boxes.

use filmcraft_time::Tick;
use serde::{Deserialize, Serialize};

/// What moves as one piece.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TextUnit {
    /// Every non-space character.
    Char,
    /// Every word (a run of non-space characters on one line).
    #[default]
    Word,
    /// Every line.
    Line,
    /// The whole text block at once.
    Whole,
}

impl TextUnit {
    pub const ALL: [TextUnit; 4] = [TextUnit::Char, TextUnit::Word, TextUnit::Line, TextUnit::Whole];
    pub fn name(self) -> &'static str {
        match self {
            TextUnit::Char => "char",
            TextUnit::Word => "word",
            TextUnit::Line => "line",
            TextUnit::Whole => "whole",
        }
    }
    pub fn from_name(s: &str) -> Option<TextUnit> {
        match s.to_ascii_lowercase().as_str() {
            "char" | "chars" | "character" | "characters" | "letter" | "letters" => Some(TextUnit::Char),
            "word" | "words" => Some(TextUnit::Word),
            "line" | "lines" => Some(TextUnit::Line),
            "whole" | "all" | "block" | "layer" => Some(TextUnit::Whole),
            _ => None,
        }
    }
}

/// In which order the units start.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum UnitOrder {
    /// Reading order (first word first).
    #[default]
    Forward,
    /// Last unit first.
    Backward,
    /// From the middle unit outwards (units equally far from the middle start together).
    CenterOut,
    /// A shuffled order (deterministic: fixed by [`TextAnim::seed`]).
    Random,
}

impl UnitOrder {
    pub const ALL: [UnitOrder; 4] = [UnitOrder::Forward, UnitOrder::Backward, UnitOrder::CenterOut, UnitOrder::Random];
    pub fn name(self) -> &'static str {
        match self {
            UnitOrder::Forward => "forward",
            UnitOrder::Backward => "backward",
            UnitOrder::CenterOut => "centerOut",
            UnitOrder::Random => "random",
        }
    }
    pub fn from_name(s: &str) -> Option<UnitOrder> {
        let s = s.to_ascii_lowercase().replace(['_', '-', ' '], "");
        UnitOrder::ALL.into_iter().find(|o| o.name().eq_ignore_ascii_case(&s)).or(match s.as_str() {
            "reverse" => Some(UnitOrder::Backward),
            "centreout" | "center" | "centre" => Some(UnitOrder::CenterOut),
            "shuffle" => Some(UnitOrder::Random),
            _ => None,
        })
    }
}

/// One phase of the animation (the entrance or the exit).
///
/// Entrance: each unit starts `offset` away from its place, `fade` transparent and at `scale` %,
/// and eases to rest over `duration`. Exit: the same in reverse (from rest to the offset /
/// transparency / scale).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TextMotion {
    /// Entrance: wait after the clip start before the first unit moves. Exit: time between the
    /// last unit's end and the clip end.
    pub delay: Tick,
    /// Time between the starts of consecutive units (0 = all together).
    pub stagger: Tick,
    /// How long each unit moves.
    pub duration: Tick,
    pub order: UnitOrder,
    /// Displacement in layer pixels `[x, y]` (y down): an entrance with `[0, 36]` rises 36 px
    /// into place; an exit with `[0, -36]` rises 36 px as it leaves.
    #[serde(deserialize_with = "lenient_pair")]
    pub offset: [f32; 2],
    /// How much the unit fades: 1 = from / to fully transparent, 0 = no fade.
    #[serde(deserialize_with = "lenient_f32")]
    pub fade: f32,
    /// Scale (%) at the far end of the move (100 = no scaling); scaled about the unit's centre.
    #[serde(deserialize_with = "lenient_f32")]
    pub scale: f32,
    /// Ease influence (0–100 %) at the start of each unit's move (0 = starts at full speed).
    #[serde(deserialize_with = "lenient_f32")]
    pub ease_in: f32,
    /// Ease influence (0–100 %) at the end of each unit's move (0 = stops abruptly).
    #[serde(deserialize_with = "lenient_f32")]
    pub ease_out: f32,
}

impl Default for TextMotion {
    fn default() -> Self {
        TextMotion {
            delay: Tick::ZERO,
            stagger: Tick::ZERO,
            duration: Tick::ZERO,
            order: UnitOrder::Forward,
            offset: [0.0, 0.0],
            fade: 1.0,
            scale: 100.0,
            ease_in: 33.3,
            ease_out: 33.3,
        }
    }
}

impl TextMotion {
    /// The motion with every number made safe (hostile project files / parameters): non-negative
    /// times, finite offsets, fade 0–1, scale 0–1000 %, influences 0–100 %.
    pub fn sanitized(&self) -> TextMotion {
        let t = |v: Tick| Tick(v.0.clamp(0, i64::MAX / 4096));
        let f = |v: f32, lo: f32, hi: f32, d: f32| if v.is_finite() { v.clamp(lo, hi) } else { d };
        TextMotion {
            delay: t(self.delay),
            stagger: t(self.stagger),
            duration: t(self.duration),
            order: self.order,
            offset: [f(self.offset[0], -100_000.0, 100_000.0, 0.0), f(self.offset[1], -100_000.0, 100_000.0, 0.0)],
            fade: f(self.fade, 0.0, 1.0, 1.0),
            scale: f(self.scale, 0.0, 1000.0, 100.0),
            ease_in: f(self.ease_in, 0.0, 100.0, 0.0),
            ease_out: f(self.ease_out, 0.0, 100.0, 0.0),
        }
    }

    /// Time from the first unit's start to the last unit's end, for `n` units.
    pub fn span(&self, n: usize) -> Tick {
        let m = self.sanitized();
        Tick(m.stagger.0.saturating_mul(ranks(n, m.order) as i64).saturating_add(m.duration.0))
    }
}

/// A number that a file may hold as `null` (JSON has no NaN): read as NaN, which
/// [`TextMotion::sanitized`] replaces with a safe value.
fn lenient_f32<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    Ok(Option::<f32>::deserialize(d)?.unwrap_or(f32::NAN))
}

fn lenient_pair<'de, D: serde::Deserializer<'de>>(d: D) -> Result<[f32; 2], D::Error> {
    let v = Option::<[Option<f32>; 2]>::deserialize(d)?.unwrap_or([None, None]);
    Ok([v[0].unwrap_or(f32::NAN), v[1].unwrap_or(f32::NAN)])
}

/// A staggered text animation (entrance and / or exit).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TextAnim {
    pub unit: TextUnit,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intro: Option<TextMotion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outro: Option<TextMotion>,
    /// Seed of the [`UnitOrder::Random`] order.
    pub seed: u32,
}

/// What one unit looks like at a moment: an offset in layer pixels, an opacity factor and a
/// scale factor (about the unit's centre).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitState {
    pub offset: [f32; 2],
    pub alpha: f32,
    pub scale: f32,
}

impl UnitState {
    pub const REST: UnitState = UnitState { offset: [0.0, 0.0], alpha: 1.0, scale: 1.0 };
    /// At rest (drawn exactly as without animation).
    pub fn is_rest(&self) -> bool {
        self.offset[0].abs() < 1e-4 && self.offset[1].abs() < 1e-4 && (self.alpha - 1.0).abs() < 1e-6 && (self.scale - 1.0).abs() < 1e-6
    }
}

/// The animation of a layer at a clip time: what [`crate::graphic::LayerSpec::anim`] carries.
#[derive(Clone, Debug, PartialEq)]
pub struct AnimClock {
    pub anim: TextAnim,
    /// Time since the clip start.
    pub time: Tick,
    /// Clip length (None = unknown: no exit is played).
    pub length: Option<Tick>,
}

/// Number of distinct start slots of `n` units in `order`, minus one (the last rank).
fn ranks(n: usize, order: UnitOrder) -> usize {
    if n == 0 {
        return 0;
    }
    match order {
        UnitOrder::CenterOut => n / 2,
        _ => n - 1,
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn hash32(i: u32, seed: u32) -> u32 {
    let mut h = i.wrapping_mul(0x9e37_79b9) ^ seed.wrapping_mul(0x85eb_ca6b) ^ 0x27d4_eb2f;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^ (h >> 15)
}

/// Start slot of unit `i` of `n` in `order`.
pub fn unit_rank(i: usize, n: usize, order: UnitOrder, seed: u32) -> usize {
    if n == 0 {
        return 0;
    }
    let i = i.min(n - 1);
    match order {
        UnitOrder::Forward => i,
        UnitOrder::Backward => n - 1 - i,
        UnitOrder::CenterOut => (2 * i).abs_diff(n - 1) / 2,
        UnitOrder::Random => {
            // an affine permutation i → (a·i + b) mod n with a coprime to n: a shuffle that costs
            // O(log n) per unit, whatever the text length
            let n128 = n as u128;
            let mut a = (hash32(1, seed) as u128 % n128).max(1);
            for _ in 0..64 {
                if gcd(a, n128) == 1 {
                    break;
                }
                a = a % n128 + 1;
            }
            if gcd(a, n128) != 1 {
                a = 1;
            }
            let b = hash32(2, seed) as u128 % n128;
            ((a * i as u128 + b) % n128) as usize
        }
    }
}

/// Cubic Bézier ease through (0,0), (`a`,0), (1−`b`,1), (1,1) — `a` / `b` are the start / end
/// influences in 0..1 — evaluated at `x` in 0..1 (bisection: deterministic, no division by a
/// possibly-zero slope).
pub fn ease(x: f32, a: f32, b: f32) -> f32 {
    let x = if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 };
    let (a, b) = (a.clamp(0.0, 1.0), b.clamp(0.0, 1.0));
    if a <= 1e-6 && b <= 1e-6 {
        return x;
    }
    let (x1, x2) = (a, 1.0 - b);
    let bez = |p1: f32, p2: f32, t: f32| {
        let u = 1.0 - t;
        3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t
    };
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..32 {
        let mid = 0.5 * (lo + hi);
        if bez(x1, x2, mid) < x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    bez(0.0, 1.0, 0.5 * (lo + hi)).clamp(0.0, 1.0)
}

/// Progress (0..1) of a move that starts at `start` and lasts `dur`, at time `t`.
fn progress(t: Tick, start: Tick, dur: Tick) -> f32 {
    if t < start {
        return 0.0;
    }
    if dur.0 <= 0 {
        return 1.0;
    }
    let done = t.0.saturating_sub(start.0);
    if done >= dur.0 { 1.0 } else { (done as f64 / dur.0 as f64) as f32 }
}

fn mul_tick(t: Tick, k: usize) -> Tick {
    Tick(t.0.saturating_mul(k.min(i64::MAX as usize) as i64))
}

/// The state of unit `i` of `n` at clip time `time` of a clip `length` long.
pub fn unit_state(anim: &TextAnim, i: usize, n: usize, time: Tick, length: Option<Tick>) -> UnitState {
    let mut st = UnitState::REST;
    if n == 0 {
        return st;
    }
    if let Some(m) = &anim.intro {
        let m = m.sanitized();
        let start = m.delay.saturating_add(mul_tick(m.stagger, unit_rank(i, n, m.order, anim.seed)));
        let e = ease(progress(time, start, m.duration), m.ease_in / 100.0, m.ease_out / 100.0);
        let k = 1.0 - e;
        st.offset = [m.offset[0] * k, m.offset[1] * k];
        st.alpha *= 1.0 - m.fade * k;
        st.scale *= 1.0 + (m.scale / 100.0 - 1.0) * k;
    }
    if let (Some(m), Some(len)) = (&anim.outro, length) {
        let m = m.sanitized();
        let first = len.saturating_sub(m.delay).saturating_sub(m.span(n));
        let start = first.saturating_add(mul_tick(m.stagger, unit_rank(i, n, m.order, anim.seed)));
        let e = ease(progress(time, start, m.duration), m.ease_in / 100.0, m.ease_out / 100.0);
        st.offset = [st.offset[0] + m.offset[0] * e, st.offset[1] + m.offset[1] * e];
        st.alpha *= 1.0 - m.fade * e;
        st.scale *= 1.0 + (m.scale / 100.0 - 1.0) * e;
    }
    st.alpha = st.alpha.clamp(0.0, 1.0);
    st.scale = st.scale.max(0.0);
    st
}

/// When each unit's entrance starts and ends (clip time), in unit order.
pub fn intro_schedule(anim: &TextAnim, n: usize) -> Vec<(Tick, Tick)> {
    let Some(m) = anim.intro.as_ref().map(TextMotion::sanitized) else { return Vec::new() };
    (0..n)
        .map(|i| {
            let s = m.delay.saturating_add(mul_tick(m.stagger, unit_rank(i, n, m.order, anim.seed)));
            (s, s.saturating_add(m.duration))
        })
        .collect()
}

/// When each unit's exit starts and ends (clip time) in a clip `length` long, in unit order.
pub fn outro_schedule(anim: &TextAnim, n: usize, length: Tick) -> Vec<(Tick, Tick)> {
    let Some(m) = anim.outro.as_ref().map(TextMotion::sanitized) else { return Vec::new() };
    let first = length.saturating_sub(m.delay).saturating_sub(m.span(n));
    (0..n)
        .map(|i| {
            let s = first.saturating_add(mul_tick(m.stagger, unit_rank(i, n, m.order, anim.seed)));
            (s, s.saturating_add(m.duration))
        })
        .collect()
}

trait SatTick {
    fn saturating_add(self, o: Tick) -> Tick;
    fn saturating_sub(self, o: Tick) -> Tick;
}
impl SatTick for Tick {
    fn saturating_add(self, o: Tick) -> Tick {
        Tick(self.0.saturating_add(o.0))
    }
    fn saturating_sub(self, o: Tick) -> Tick {
        Tick(self.0.saturating_sub(o.0))
    }
}

/// Ticks of `frames` frames at 60 fps (presets are authored at 60 fps and keep their real-time
/// length at every sequence rate).
pub fn frames60(frames: i64) -> Tick {
    Tick(frames.saturating_mul(filmcraft_time::TICKS_PER_SECOND / 60))
}

/// Built-in animation presets (original FilmCraft settings): `(id, description, animation)`.
pub fn presets() -> Vec<(&'static str, &'static str, TextAnim)> {
    let rise = |stagger: i64, dur: i64, rise: f32| TextMotion {
        stagger: frames60(stagger),
        duration: frames60(dur),
        offset: [0.0, rise],
        ease_in: 60.0,
        ease_out: 80.0,
        ..Default::default()
    };
    let fade_out = TextMotion { duration: frames60(18), ease_in: 33.3, ease_out: 33.3, ..Default::default() };
    vec![
        (
            "word-rise",
            "Words rise 36 px and fade in one after another (14-frame offset, 20 frames each at 60 fps: the phrase lands in 48 frames); all fade out together over the last 18 frames.",
            TextAnim { unit: TextUnit::Word, intro: Some(rise(14, 20, 36.0)), outro: Some(fade_out.clone()), seed: 0 },
        ),
        (
            "char-rise",
            "Letters rise 24 px and fade in one by one (2-frame offset, 14 frames each at 60 fps); all fade out together over the last 18 frames.",
            TextAnim { unit: TextUnit::Char, intro: Some(rise(2, 14, 24.0)), outro: Some(fade_out.clone()), seed: 0 },
        ),
        (
            "line-rise",
            "Lines rise 48 px and fade in one after another (12-frame offset, 30 frames each at 60 fps); all fade out together over the last 18 frames.",
            TextAnim { unit: TextUnit::Line, intro: Some(rise(12, 30, 48.0)), outro: Some(fade_out.clone()), seed: 0 },
        ),
        (
            "whole-rise",
            "The whole block rises 48 px and fades in over 36 frames at 60 fps (smooth ease in and out); fades out over the last 18 frames.",
            TextAnim {
                unit: TextUnit::Whole,
                intro: Some(TextMotion { duration: frames60(36), offset: [0.0, 48.0], ..Default::default() }),
                outro: Some(fade_out),
                seed: 0,
            },
        ),
    ]
}

/// A built-in preset by id.
pub fn preset(id: &str) -> Option<TextAnim> {
    presets().into_iter().find(|(k, _, _)| k.eq_ignore_ascii_case(id)).map(|(_, _, a)| a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word_rise() -> TextAnim {
        preset("word-rise").unwrap()
    }

    #[test]
    fn units_enter_one_after_another() {
        let a = word_rise();
        let f = frames60;
        // before anything: all three words hidden 36 px below
        for i in 0..3 {
            let s = unit_state(&a, i, 3, Tick::ZERO, None);
            assert_eq!(s.alpha, 0.0);
            assert!((s.offset[1] - 36.0).abs() < 1e-4);
        }
        // at frame 14 the first word is done moving? no: it lasts 20 frames; the second starts now
        let s0 = unit_state(&a, 0, 3, f(14), None);
        let s1 = unit_state(&a, 1, 3, f(14), None);
        let s2 = unit_state(&a, 2, 3, f(14), None);
        assert!(s0.alpha > 0.5 && s0.offset[1] < 18.0, "{s0:?}");
        assert_eq!(s1.alpha, 0.0);
        assert_eq!(s2.alpha, 0.0);
        // the whole phrase lands at frame 48 and holds
        for i in 0..3 {
            assert!(unit_state(&a, i, 3, f(48), None).is_rest());
            assert!(unit_state(&a, i, 3, f(200), Some(f(300))).is_rest());
        }
        // exit: all fade together over the last 18 frames, gone at the end
        let mid = unit_state(&a, 1, 3, f(291), Some(f(300)));
        assert!(mid.alpha > 0.2 && mid.alpha < 0.8 && mid.offset[1].abs() < 1e-4, "{mid:?}");
        assert_eq!(unit_state(&a, 2, 3, f(300), Some(f(300))).alpha, 0.0);
        assert_eq!(intro_schedule(&a, 3), vec![(f(0), f(20)), (f(14), f(34)), (f(28), f(48))]);
        assert_eq!(outro_schedule(&a, 3, f(300)), vec![(f(282), f(300)); 3]);
    }

    #[test]
    fn ease_is_monotonic_and_matches_ends() {
        for (a, b) in [(0.0, 0.0), (0.6, 0.8), (1.0, 1.0), (0.333, 0.333)] {
            let mut prev = 0.0;
            for k in 0..=100 {
                let y = ease(k as f32 / 100.0, a, b);
                assert!(y + 1e-5 >= prev, "{a} {b} {k}");
                prev = y;
            }
            assert!(ease(0.0, a, b).abs() < 1e-5 && (ease(1.0, a, b) - 1.0).abs() < 1e-5);
        }
        // a symmetric 1/3 ease is close to smoothstep (the "ease in / ease out" keyframe curve)
        let s = |x: f32| x * x * (3.0 - 2.0 * x);
        assert!((ease(0.25, 0.333, 0.333) - s(0.25)).abs() < 0.03);
    }

    #[test]
    fn orders() {
        let r = |o| (0..5).map(|i| unit_rank(i, 5, o, 7)).collect::<Vec<_>>();
        assert_eq!(r(UnitOrder::Forward), vec![0, 1, 2, 3, 4]);
        assert_eq!(r(UnitOrder::Backward), vec![4, 3, 2, 1, 0]);
        assert_eq!(r(UnitOrder::CenterOut), vec![2, 1, 0, 1, 2]);
        let mut rnd = r(UnitOrder::Random);
        assert_eq!(rnd, r(UnitOrder::Random), "deterministic");
        rnd.sort();
        assert_eq!(rnd, vec![0, 1, 2, 3, 4], "a permutation");
        assert_eq!(UnitOrder::from_name("center_out"), Some(UnitOrder::CenterOut));
        assert_eq!(TextUnit::from_name("letters"), Some(TextUnit::Char));
    }

    #[test]
    fn hostile_values_do_not_panic() {
        let m = TextMotion {
            delay: Tick(i64::MIN),
            stagger: Tick(i64::MAX),
            duration: Tick(-5),
            offset: [f32::NAN, f32::INFINITY],
            fade: f32::NAN,
            scale: -4.0,
            ease_in: 1e9,
            ease_out: f32::NEG_INFINITY,
            order: UnitOrder::Random,
        };
        let a = TextAnim { unit: TextUnit::Char, intro: Some(m.clone()), outro: Some(m), seed: u32::MAX };
        for i in [0usize, 1, usize::MAX] {
            for t in [i64::MIN, -1, 0, 1, i64::MAX] {
                let s = unit_state(&a, i, usize::MAX, Tick(t), Some(Tick(t)));
                assert!(s.alpha.is_finite() && s.scale.is_finite() && s.offset.iter().all(|v| v.is_finite()));
            }
        }
        assert_eq!(unit_state(&a, 0, 0, Tick(0), None), UnitState::REST);
        // NaN is written as null; reading it back must not fail (a damaged project) and the
        // values come back sanitised
        let json = serde_json::to_string(&a).unwrap();
        let back: TextAnim = serde_json::from_str(&json).unwrap();
        assert_eq!(back.unit, TextUnit::Char);
        let m = back.intro.unwrap().sanitized();
        assert!(m.offset.iter().all(|v| v.is_finite()) && m.fade == 1.0);
    }
}
