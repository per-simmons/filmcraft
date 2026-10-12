//! Word-aware text commands for agents: staggered text animation, per-word layout boxes and
//! emphasis titles with balanced lines.
//!
//! - `graphics.animateText` stores a staggered entrance / exit on a text layer (by character,
//!   word, line or the whole block: offset between units, per-unit duration, rise, fade, scale,
//!   easing). It needs no keyframes and follows text edits; see
//!   [`filmcraft_project::text_anim`].
//! - `graphics.textLayout` returns the rectangle of every word / character / line of a text layer
//!   in sequence pixels at a time (at rest, or as animated), plus line widths and a balance
//!   figure, so an agent can place a highlight or check line breaks without looking.
//! - `graphics.emphasisTitle` makes a white ALL CAPS title in one step: balanced line breaks, a
//!   placement (lower centre, upper left / right, centre, left / right third), the serif look and
//!   the word-rise animation.

use filmcraft_geom::{Affine, Vec2};
use filmcraft_project::graphic::{self, LayerContent, LayerSpec, TextProps, eval_layer};
use filmcraft_project::text_anim::{self, TextAnim, TextMotion, TextUnit, UnitOrder, unit_state};
use filmcraft_project::{ClipId, ItemKind, ParamValue};
use filmcraft_render::graphic_clip::{item_layer_specs, layer_matrix, text_layout, text_units, unit_matrix};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, f64_p, has_seq, str_p, time_p};
use crate::graphics::{layer_effect_index, target_clip};
use crate::{EngineError, Result, Session};

fn has_graphic(s: &Session) -> std::result::Result<(), String> {
    has_seq(s)?;
    target_clip(s, &Value::Null).map(|_| ()).ok_or_else(|| "select a graphic clip".into())
}

/// A duration from `<key>Frames` (sequence frames), `<key>Seconds` or `<key>` (ticks).
fn dur_p(s: &Session, p: &Value, key: &str) -> Option<Tick> {
    if let Some(f) = f64_p(p, &format!("{key}Frames")).filter(|f| f.is_finite()) {
        return Some(s.sequence_rate().tick_of(f.round().clamp(0.0, 1e9) as i64));
    }
    if let Some(sec) = f64_p(p, &format!("{key}Seconds")).filter(|f| f.is_finite()) {
        return Some(Tick::from_seconds_f64(sec.clamp(0.0, 1e6)));
    }
    p.get(key).and_then(Value::as_i64).map(|t| Tick(t.clamp(0, i64::MAX / 4096)))
}

fn finite(p: &Value, k: &str) -> Option<f32> {
    f64_p(p, k).filter(|v| v.is_finite()).map(|v| v.clamp(-1e6, 1e6) as f32)
}

/// Fade from a bool (on = full) or a percentage.
fn fade_p(p: &Value) -> Option<f32> {
    match p.get("fade") {
        Some(Value::Bool(b)) => Some(if *b { 1.0 } else { 0.0 }),
        Some(v) => v.as_f64().filter(|v| v.is_finite()).map(|v| (v / 100.0).clamp(0.0, 1.0) as f32),
        None => None,
    }
}

/// Apply the motion fields of `p` onto `m`. `exit` flips the meaning of `rise` (an exit rises by
/// moving up, an entrance by starting below).
fn motion_from(s: &Session, p: &Value, mut m: TextMotion, exit: bool, n_units: usize, cmd: &str) -> Result<TextMotion> {
    if let Some(d) = dur_p(s, p, "delay") {
        m.delay = d;
    }
    if let Some(d) = dur_p(s, p, "duration") {
        m.duration = d;
    }
    if let Some(d) = dur_p(s, p, "stagger") {
        m.stagger = d;
    } else if let Some(total) = dur_p(s, p, "total") {
        // the whole entrance lasts `total`: spread the starts so the last unit ends then
        if dur_p(s, p, "duration").is_none() && m.duration > total {
            m.duration = total;
        }
        let slots = if n_units > 1 { (n_units - 1) as i64 } else { 1 };
        m.stagger = Tick((total.0.saturating_sub(m.duration.0)).max(0) / slots);
    }
    if let Some(o) = str_p(p, "order") {
        m.order = UnitOrder::from_name(o).ok_or_else(|| bad(cmd, format!("unknown order `{o}` (forward, backward, centerOut, random)")))?;
    }
    if let Some(r) = finite(p, "rise") {
        m.offset[1] = if exit { -r } else { r };
    }
    if let Some(x) = finite(p, "driftX") {
        m.offset[0] = x;
    }
    if let Some(f) = fade_p(p) {
        m.fade = f;
    }
    if let Some(sc) = finite(p, "scale") {
        m.scale = sc.clamp(0.0, 1000.0);
    }
    if let Some(e) = finite(p, "easeIn") {
        m.ease_in = e.clamp(0.0, 100.0);
    }
    if let Some(e) = finite(p, "easeOut") {
        m.ease_out = e.clamp(0.0, 100.0);
    }
    Ok(m.sanitized())
}

/// The clip, its graphic canvas and the layer's effect index; errors unless it is a text layer.
fn text_target(s: &Session, p: &Value, cmd: &str) -> Result<(ClipId, usize, usize, (u32, u32))> {
    let clip = target_clip(s, p).ok_or_else(|| bad(cmd, "no graphic clip"))?;
    let (layer, ei) = layer_effect_index(s, clip, p)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, it) = seq.find_item(clip).ok_or_else(|| bad(cmd, "no such clip"))?;
    let canvas = match s.project.item(it.item).map(|p| &p.kind) {
        Some(ItemKind::Graphic { width, height, .. }) => (*width, *height),
        _ => (seq.settings.width, seq.settings.height),
    };
    let e = it.effects.get(ei).ok_or_else(|| bad(cmd, "no such layer"))?;
    if e.effect != graphic::TEXT_LAYER {
        return Err(bad(cmd, "not a text layer"));
    }
    Ok((clip, layer, ei, canvas))
}

/// The layer as shown at timeline time `tl` (pins, rolls, animation clock applied).
fn shown_layer(s: &Session, clip: ClipId, ei: usize, canvas: (u32, u32), tl: Tick) -> Result<(LayerSpec, Tick, Tick)> {
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, it) = seq.find_item(clip).ok_or_else(|| bad("graphics", "no such clip"))?;
    let tl = tl.clamp(it.start, (it.end() - Tick(1)).max(it.start));
    let mt = it.source_time_at(tl);
    let spec = item_layer_specs(it, mt, canvas).into_iter().find(|(x, _)| *x == ei).map(|(_, s)| s).ok_or_else(|| bad("graphics", "no such layer"))?;
    Ok((spec, mt, tl))
}

/// Unit count of a layer's text for `unit`.
fn unit_count(spec: &LayerSpec, unit: TextUnit) -> usize {
    match &spec.content {
        LayerContent::Text(t) => text_units(&text_layout(t), &t.text, unit).len(),
        LayerContent::Shape(_) => 0,
    }
}

fn animate_text(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "graphics.animateText";
    let (clip, layer, ei, canvas) = text_target(s, p, ID)?;
    let ph = s.playhead();
    let (spec, _, _) = shown_layer(s, clip, ei, canvas, ph)?;
    let current = spec.anim.as_ref().map(|c| c.anim.clone());
    let off = bool_p(p, "off") == Some(true) || p.get("preset").and_then(Value::as_str).is_some_and(|x| x.eq_ignore_ascii_case("none"));
    let anim = if off {
        None
    } else {
        let mut a = match str_p(p, "preset") {
            Some(name) => text_anim::preset(name).ok_or_else(|| {
                let names: Vec<&str> = text_anim::presets().into_iter().map(|(k, _, _)| k).collect();
                bad(ID, format!("unknown preset `{name}` ({})", names.join(", ")))
            })?,
            None => current.unwrap_or_else(|| text_anim::preset("word-rise").unwrap_or_default()),
        };
        if let Some(u) = str_p(p, "unit") {
            a.unit = TextUnit::from_name(u).ok_or_else(|| bad(ID, format!("unknown unit `{u}` (char, word, line, whole)")))?;
        }
        if let Some(seed) = p.get("seed").and_then(Value::as_u64) {
            a.seed = seed.min(u32::MAX as u64) as u32;
        }
        let n = unit_count(&spec, a.unit);
        // intro fields may sit at the top level or in `intro`
        match p.get("intro") {
            Some(Value::Bool(false)) | Some(Value::Null) => a.intro = None,
            Some(v @ Value::Object(_)) => a.intro = Some(motion_from(s, v, a.intro.clone().unwrap_or_default(), false, n, ID)?),
            _ => {
                let keys = ["delay", "duration", "stagger", "total", "order", "rise", "driftX", "fade", "scale", "easeIn", "easeOut"];
                let touched = p.as_object().is_some_and(|o| o.keys().any(|k| keys.iter().any(|x| k == x || k.starts_with(x))));
                if touched {
                    a.intro = Some(motion_from(s, p, a.intro.clone().unwrap_or_default(), false, n, ID)?);
                }
            }
        }
        match p.get("outro") {
            Some(Value::Bool(false)) | Some(Value::Null) => a.outro = None,
            Some(Value::Bool(true)) => {
                a.outro = Some(a.outro.clone().unwrap_or(TextMotion { duration: text_anim::frames60(18), ..Default::default() }));
            }
            Some(v @ Value::Object(_)) => {
                let base = a.outro.clone().unwrap_or(TextMotion { duration: text_anim::frames60(18), ..Default::default() });
                a.outro = Some(motion_from(s, v, base, true, n, ID)?);
            }
            _ => {}
        }
        if a.intro.is_none() && a.outro.is_none() { None } else { Some(a) }
    };
    let stored = anim.clone();
    s.edit_sequence("Animate Text", move |q, _, _| {
        let (_, it) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
        let e = it.effects.get_mut(ei).ok_or_else(|| bad(ID, "no such layer"))?;
        let x = e.layer.get_or_insert_with(Default::default);
        x.anim = stored;
        if x.is_empty() {
            e.layer = None;
        }
        Ok(())
    })?;
    let mut out = schedule_json(s, clip, ei, canvas)?;
    if let Some(o) = out.as_object_mut() {
        o.insert("layer".into(), json!(layer));
    }
    Ok(out)
}

/// The stored animation of a layer and when each unit moves (clip-relative and sequence ticks).
fn schedule_json(s: &Session, clip: ClipId, ei: usize, canvas: (u32, u32)) -> Result<Value> {
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let rate = seq.settings.frame_rate;
    let (_, it) = seq.find_item(clip).ok_or_else(|| bad("graphics.animateText", "no such clip"))?;
    let (spec, _, _) = shown_layer(s, clip, ei, canvas, it.start)?;
    let anim = it.effects.get(ei).and_then(|e| e.layer.as_ref()).and_then(|x| x.anim.clone());
    let Some(a) = anim else { return Ok(json!({"clip": clip.0, "animation": null})) };
    let LayerContent::Text(t) = &spec.content else { return Ok(json!({"clip": clip.0, "animation": null})) };
    let units = text_units(&text_layout(t), &t.text, a.unit);
    let n = units.len();
    let len = Tick((it.duration.0 as f64 * it.speed.abs().max(1e-9)).round() as i64);
    let intro = text_anim::intro_schedule(&a, n);
    let outro = text_anim::outro_schedule(&a, n, len);
    let frame = |t: Tick| rate.frame_at(t);
    let rows: Vec<Value> = units
        .iter()
        .enumerate()
        .map(|(i, u)| {
            let w = |x: Option<&(Tick, Tick)>| {
                x.map(|(a, b)| json!({"start": a.0, "end": b.0, "seqStart": (it.start + *a).0, "seqEnd": (it.start + *b).0, "seqStartFrame": frame(it.start + *a), "seqEndFrame": frame(it.start + *b)}))
            };
            json!({"index": i, "text": t.text.get(u.range.clone()).unwrap_or_default(), "line": u.line, "intro": w(intro.get(i)), "outro": w(outro.get(i))})
        })
        .collect();
    let intro_end = intro.iter().map(|x| x.1).max();
    let outro_start = outro.iter().map(|x| x.0).min();
    Ok(json!({
        "clip": clip.0,
        "animation": serde_json::to_value(&a).unwrap_or_default(),
        "unitCount": n,
        "units": rows,
        "introEnd": intro_end.map(|t| t.0),
        "introEndSeq": intro_end.map(|t| (it.start + t).0),
        "outroStart": outro_start.map(|t| t.0),
        "outroStartSeq": outro_start.map(|t| (it.start + t).0),
        "clipStart": it.start.0,
        "clipDuration": it.duration.0,
    }))
}

/// Axis-aligned bounds of a layer-pixel rectangle mapped through `m`, as `[x, y, w, h]`.
fn map_rect(m: &Affine, r: [f32; 4]) -> [f64; 4] {
    let pts = [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])].map(|(x, y)| m.apply(Vec2::new(x as f64, y as f64)));
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in pts {
        x0 = x0.min(p.x);
        y0 = y0.min(p.y);
        x1 = x1.max(p.x);
        y1 = y1.max(p.y);
    }
    let r2 = |v: f64| (v * 100.0).round() / 100.0;
    [r2(x0), r2(y0), r2(x1 - x0), r2(y1 - y0)]
}

fn text_layout_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "graphics.textLayout";
    let (clip, layer, ei, canvas) = text_target(s, p, ID)?;
    let unit = match str_p(p, "unit") {
        Some(u) => TextUnit::from_name(u).ok_or_else(|| bad(ID, format!("unknown unit `{u}` (char, word, line, whole)")))?,
        None => TextUnit::Word,
    };
    let animated = bool_p(p, "animated").unwrap_or(false);
    let tl = time_p(s, p, "").unwrap_or_else(|| s.playhead());
    let (spec, mt, tl) = shown_layer(s, clip, ei, canvas, tl)?;
    let LayerContent::Text(t) = &spec.content else { return Err(bad(ID, "not a text layer")) };
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, it) = seq.find_item(clip).ok_or_else(|| bad(ID, "no such clip"))?;
    // graphic canvas → sequence pixels (the clip's Motion), after layer → canvas
    let motion = filmcraft_render::motion_matrix(seq, it, canvas, mt);
    let lm = motion.then_apply(&layer_matrix(&spec));
    let l = text_layout(t);
    let units = text_units(&l, &t.text, unit);
    let n = units.len();
    let clock = spec.anim.as_ref();
    let rows: Vec<Value> = units
        .iter()
        .enumerate()
        .map(|(i, u)| {
            let st = match (animated, clock) {
                (true, Some(c)) => text_anim_state(&c.anim, unit, i, n, c, &l, &t.text),
                _ => text_anim::UnitState::REST,
            };
            let m = lm.then_apply(&unit_matrix(u.rect, &st));
            let r = map_rect(&m, u.rect);
            let bl = m.apply(Vec2::new(u.rect[0] as f64, u.baseline as f64));
            json!({
                "index": i,
                "text": t.text.get(u.range.clone()).unwrap_or_default(),
                "line": u.line,
                "rect": r,
                "center": [r[0] + r[2] / 2.0, r[1] + r[3] / 2.0],
                "baselineY": (bl.y * 100.0).round() / 100.0,
                "opacity": st.alpha,
                "chars": [u.range.start, u.range.end],
            })
        })
        .collect();
    let line_units = text_units(&l, &t.text, TextUnit::Line);
    let lines: Vec<Value> = line_units
        .iter()
        .map(|u| {
            let r = map_rect(&lm, u.rect);
            json!({"index": u.line, "text": t.text.get(u.range.clone()).unwrap_or_default(), "rect": r, "width": r[2]})
        })
        .collect();
    let widths: Vec<f64> = lines.iter().filter_map(|l| l["width"].as_f64()).collect();
    let (wmin, wmax) = widths.iter().fold((f64::MAX, 0.0f64), |(a, b), w| (a.min(*w), b.max(*w)));
    let balance = if widths.len() > 1 && wmax > 0.0 { wmin / wmax } else { 1.0 };
    // a last line holding one word that is much shorter than the others
    let words_per_line: Vec<usize> = (0..line_units.len()).map(|li| units_on_line(&l, &t.text, li)).collect();
    let orphan = line_units.len() > 1 && words_per_line.last() == Some(&1) && widths.last().is_some_and(|w| *w < wmax * 0.6);
    let block = map_rect(&lm, l.bounds);
    let (fw, fh) = (seq.settings.width as f64, seq.settings.height as f64);
    let inside = |r: [f64; 4], m: f64| r[0] >= fw * m && r[1] >= fh * m && r[0] + r[2] <= fw * (1.0 - m) && r[1] + r[3] <= fh * (1.0 - m);
    Ok(json!({
        "clip": clip.0,
        "layer": layer,
        "unit": unit.name(),
        "time": tl.0,
        "animated": animated,
        "text": t.text,
        "frame": [seq.settings.width, seq.settings.height],
        "canvas": [canvas.0, canvas.1],
        "units": rows,
        "lines": lines,
        "block": block,
        "balance": {"lineWidths": widths, "ratio": (balance * 1000.0).round() / 1000.0, "orphan": orphan},
        "safe": {"insideFrame": inside(block, 0.0), "actionSafe": inside(block, 0.05), "titleSafe": inside(block, 0.1)},
        "fontMissing": l.missing_font,
    }))
}

fn units_on_line(l: &filmcraft_text::Layout, text: &str, line: usize) -> usize {
    text_units(l, text, TextUnit::Word).iter().filter(|u| u.line == line).count()
}

/// The animated state of unit `i` when the query's unit differs from the animation's unit: the
/// state of the animation unit that holds this unit's first character.
fn text_anim_state(a: &TextAnim, unit: TextUnit, i: usize, n: usize, c: &text_anim::AnimClock, l: &filmcraft_text::Layout, text: &str) -> text_anim::UnitState {
    if unit == a.unit {
        return unit_state(a, i, n, c.time, c.length);
    }
    let mine = text_units(l, text, unit);
    let theirs = text_units(l, text, a.unit);
    let Some(start) = mine.get(i).map(|u| u.range.start) else { return text_anim::UnitState::REST };
    match theirs.iter().position(|u| u.range.start <= start && start < u.range.end.max(u.range.start + 1)) {
        Some(k) => unit_state(a, k, theirs.len(), c.time, c.length),
        None => text_anim::UnitState::REST,
    }
}

// ------------------------------------------------------------------------------------------------
// Emphasis titles
// ------------------------------------------------------------------------------------------------

/// Where an emphasis title sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    LowerCenter,
    UpperLeft,
    UpperRight,
    Center,
    LeftThird,
    RightThird,
}

impl Placement {
    fn from_name(s: &str) -> Option<Placement> {
        Some(match s.to_ascii_lowercase().replace(['_', '-', ' '], "").as_str() {
            "lowercenter" | "lowercentre" | "bottom" | "lower" => Placement::LowerCenter,
            "upperleft" | "topleft" => Placement::UpperLeft,
            "upperright" | "topright" => Placement::UpperRight,
            "center" | "centre" | "middle" => Placement::Center,
            "leftthird" | "left" => Placement::LeftThird,
            "rightthird" | "right" => Placement::RightThird,
            _ => return None,
        })
    }
    fn name(self) -> &'static str {
        match self {
            Placement::LowerCenter => "lowerCenter",
            Placement::UpperLeft => "upperLeft",
            Placement::UpperRight => "upperRight",
            Placement::Center => "center",
            Placement::LeftThird => "leftThird",
            Placement::RightThird => "rightThird",
        }
    }
    /// 0 left, 1 centre, 2 right.
    fn align(self) -> u32 {
        match self {
            Placement::UpperLeft => 0,
            Placement::UpperRight => 2,
            _ => 1,
        }
    }
    /// Default widest line, as a fraction of the frame width.
    fn max_width(self) -> f64 {
        match self {
            Placement::LowerCenter | Placement::Center => 0.84,
            Placement::UpperLeft | Placement::UpperRight => 0.42,
            Placement::LeftThird | Placement::RightThird => 0.40,
        }
    }
}

/// The emphasis serif: Bodoni 72 Bold when the system has it, else the bundled Noto Serif
/// (OFL) in bold. Returns (family, style, fell back).
fn emphasis_font() -> (String, String, bool) {
    let r = filmcraft_text::fonts::resolve("Bodoni 72", "Bold");
    if !r.missing {
        return ("Bodoni 72".into(), "Bold".into(), false);
    }
    ("Noto Serif".into(), "Bold".into(), true)
}

/// Split `words` into at most `max_lines` lines so that the widest line is as narrow as possible,
/// using the fewest lines whose widest line fits `max_w` (all `max_lines` when none fits).
/// `width(a, b)` measures words `a..b` set on one line. Returns (break indices, widest, fits).
fn balance_lines(n: usize, max_lines: usize, max_w: f64, width: &mut dyn FnMut(usize, usize) -> f64) -> (Vec<usize>, f64, bool) {
    if n == 0 {
        return (Vec::new(), 0.0, true);
    }
    let max_lines = max_lines.clamp(1, n.min(8));
    let mut w = vec![vec![0.0f64; n + 1]; n + 1];
    for a in 0..n {
        for b in a + 1..=n {
            if let Some(row) = w.get_mut(a)
                && let Some(c) = row.get_mut(b)
            {
                *c = width(a, b);
            }
        }
    }
    let wid = |a: usize, b: usize| w.get(a).and_then(|r| r.get(b)).copied().unwrap_or(f64::MAX);
    let mut last = (Vec::new(), f64::MAX, false);
    for k in 1..=max_lines {
        // best[j][i]: the narrowest widest line setting the first i words on j lines
        let mut best = vec![vec![(f64::MAX, 0usize); n + 1]; k + 1];
        if let Some(c) = best.get_mut(0).and_then(|r| r.get_mut(0)) {
            *c = (0.0, 0);
        }
        for j in 1..=k {
            for i in j..=n {
                let mut cell = (f64::MAX, 0usize);
                for a in (j - 1)..i {
                    let prev = best.get(j - 1).and_then(|r| r.get(a)).map_or(f64::MAX, |c| c.0);
                    // ties go to the later break: a longer last line, never a stranded word
                    let cand = prev.max(wid(a, i));
                    if cand <= cell.0 {
                        cell = (cand, a);
                    }
                }
                if let Some(c) = best.get_mut(j).and_then(|r| r.get_mut(i)) {
                    *c = cell;
                }
            }
        }
        let widest = best.get(k).and_then(|r| r.get(n)).map_or(f64::MAX, |c| c.0);
        let mut breaks = Vec::new();
        let mut i = n;
        for j in (1..=k).rev() {
            let a = best.get(j).and_then(|r| r.get(i)).map_or(0, |c| c.1);
            if j > 1 {
                breaks.push(a);
            }
            i = a;
        }
        breaks.reverse();
        last = (breaks, widest, widest <= max_w);
        if widest <= max_w {
            return last;
        }
    }
    last
}

fn emphasis_title(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "graphics.emphasisTitle";
    let raw = str_p(p, "text").ok_or_else(|| bad(ID, "need `text`"))?;
    let words: Vec<String> = raw.split_whitespace().map(|w| if bool_p(p, "caps") == Some(false) { w.to_string() } else { w.to_uppercase() }).collect();
    if words.is_empty() {
        return Err(bad(ID, "`text` is empty"));
    }
    if words.len() > 64 {
        return Err(bad(ID, "an emphasis title holds at most 64 words"));
    }
    let placement = match str_p(p, "placement") {
        Some(x) => Placement::from_name(x)
            .ok_or_else(|| bad(ID, format!("unknown placement `{x}` (lowerCenter, upperLeft, upperRight, center, leftThird, rightThird)")))?,
        None => Placement::LowerCenter,
    };
    let (fw, fh) = s.active_sequence().map(|q| (q.settings.width as f64, q.settings.height as f64)).ok_or(EngineError::NoSequence)?;
    let k = fh / 2160.0;
    let size = f64_p(p, "size").filter(|v| v.is_finite()).unwrap_or(222.0 * k).clamp(4.0, 2000.0);
    let tracking = f64_p(p, "tracking").filter(|v| v.is_finite()).unwrap_or(15.0).clamp(-1000.0, 10_000.0);
    let (font, style, fell_back) = match str_p(p, "font") {
        Some(f) => (f.to_string(), str_p(p, "fontStyle").unwrap_or("Bold").to_string(), false),
        None => emphasis_font(),
    };
    let max_lines = p.get("maxLines").and_then(Value::as_u64).unwrap_or(3).clamp(1, 8) as usize;
    let max_w = f64_p(p, "maxWidth").filter(|v| v.is_finite() && *v > 0.0).unwrap_or(placement.max_width() * fw);
    let color = p.get("color").cloned().unwrap_or(json!("#ffffff"));

    // measure with the real text settings
    let mut base = graphic::new_text_layer("", Vec2::new(0.0, 0.0), size);
    let set = |e: &mut filmcraft_project::EffectInstance, k: &str, v: ParamValue| {
        e.params.insert(k.into(), filmcraft_project::Param::new(v));
    };
    set(&mut base, "font", ParamValue::Text(font.clone()));
    set(&mut base, "font_style", ParamValue::Text(style.clone()));
    set(&mut base, "tracking", ParamValue::Float(tracking));
    set(&mut base, "align", ParamValue::Choice(placement.align()));
    let props: TextProps = match eval_layer(&base, Tick::ZERO, (fw as u32, fh as u32)).map(|l| l.content) {
        Some(LayerContent::Text(t)) => t,
        _ => return Err(bad(ID, "could not set up the text")),
    };
    let mut measure = |a: usize, b: usize| {
        let line = words.get(a..b).map(|w| w.join(" ")).unwrap_or_default();
        let l = text_layout(&TextProps { text: line, ..props.clone() });
        l.lines.first().map_or(0.0, |x| x.width as f64)
    };
    let (breaks, widest, fits) = balance_lines(words.len(), max_lines, max_w, &mut measure);
    let mut lines: Vec<String> = Vec::new();
    let mut a = 0;
    for b in breaks.iter().copied().chain(std::iter::once(words.len())) {
        lines.push(words.get(a..b).map(|w| w.join(" ")).unwrap_or_default());
        a = b;
    }
    let text = lines.join("\n");
    let l = text_layout(&TextProps { text: text.clone(), ..props.clone() });
    let first_bl = l.lines.first().map_or(0.0, |x| x.baseline) as f64;
    let last_bl = l.lines.last().map_or(0.0, |x| x.baseline) as f64;
    let (top, bottom) = (l.bounds[1] as f64, l.bounds[3] as f64);
    // the point-text origin: the alignment point on the first baseline
    let pos = match placement {
        // one line sits with its baseline at 80 % of the height (1728 of 2160); more lines grow up
        Placement::LowerCenter => Vec2::new(fw / 2.0, fh * 0.8 - (last_bl - first_bl)),
        Placement::UpperLeft => Vec2::new(fw * 0.07, fh * 0.17 - top),
        Placement::UpperRight => Vec2::new(fw * 0.93, fh * 0.17 - top),
        Placement::Center => Vec2::new(fw / 2.0, fh / 2.0 - (top + bottom) / 2.0),
        Placement::LeftThird => Vec2::new(fw * 0.25, fh / 2.0 - (top + bottom) / 2.0),
        Placement::RightThird => Vec2::new(fw * 0.75, fh / 2.0 - (top + bottom) / 2.0),
    };
    let seconds = f64_p(p, "seconds").filter(|v| v.is_finite()).unwrap_or(5.0).clamp(0.1, 3600.0);
    let mut q = json!({"text": text, "position": [pos.x, pos.y], "size": size, "font": font, "fontStyle": style, "seconds": seconds, "newClip": true});
    for k in ["time", "frame", "timecode", "track"] {
        if let Some(v) = p.get(k) {
            q[k] = v.clone();
        }
    }
    let before = s.history.undo.len();
    let r = s.execute("graphics.newText", q)?;
    let clip = r.get("clip").and_then(Value::as_u64).map(ClipId).ok_or_else(|| bad(ID, "no clip made"))?;
    let finish = |s: &mut Session| -> Result<Value> {
        s.execute(
            "graphics.set",
            json!({"clip": clip.0, "layer": 0, "props": {"tracking": tracking, "align": placement.align(), "fill": true, "fill_color": color, "background": false, "stroke": false, "shadow": false, "rotation": 0, "name": "Emphasis"}}),
        )?;
        let anim = match p.get("animate") {
            Some(Value::Bool(false)) => None,
            Some(Value::String(x)) => Some(x.clone()),
            Some(Value::Object(_)) => Some(String::new()),
            _ => Some("word-rise".to_string()),
        };
        if let Some(name) = anim {
            let mut ap = match p.get("animate") {
                Some(o @ Value::Object(_)) => o.clone(),
                _ => json!({"preset": name}),
            };
            ap["clip"] = json!(clip.0);
            ap["layer"] = json!(0);
            if ap.get("preset").is_none() && str_p(&ap, "unit").is_none() {
                ap["preset"] = json!("word-rise");
            }
            s.execute("graphics.animateText", ap)?;
        }
        Ok(Value::Null)
    };
    if let Err(e) = finish(s) {
        while s.history.undo.len() > before {
            if s.execute("edit.undo", json!({})).is_err() {
                break;
            }
        }
        return Err(e);
    }
    // one undo step
    if s.history.undo.len() > before + 1 {
        s.history.undo.truncate(before + 1);
    }
    if let Some(e) = s.history.undo.get_mut(before) {
        e.0 = "Emphasis Title".to_string();
    }
    let start = s.active_sequence().and_then(|q| q.find_item(clip)).map(|(_, it)| it.start).unwrap_or(Tick::ZERO);
    let dur = s.active_sequence().and_then(|q| q.find_item(clip)).map(|(_, it)| it.duration).unwrap_or(Tick::ZERO);
    // the resting layout (mid-hold), so the caller can check placement without rendering
    let hold = start + Tick(dur.0 / 2);
    let layout = text_layout_cmd(s, &json!({"clip": clip.0, "layer": 0, "unit": "word", "time": hold.0}))?;
    let (_, ei) = layer_effect_index(s, clip, &json!({"layer": 0}))?;
    let schedule = schedule_json(s, clip, ei, (fw as u32, fh as u32))?;
    Ok(json!({
        "clip": clip.0,
        "layer": 0,
        "text": text,
        "lines": lines,
        "placement": placement.name(),
        "font": font,
        "fontStyle": style,
        "fontFallback": fell_back,
        "size": size,
        "tracking": tracking,
        "position": [pos.x, pos.y],
        "widestLine": widest,
        "maxWidth": max_w,
        "fits": fits,
        "layout": layout,
        "animation": schedule,
    }))
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "graphics.animateText",
            label: "Animate Text",
            menu: &[],
            shortcut: None,
            params: r#"{"clip":id?,"layer":n|name?,"preset":"word-rise|char-rise|line-rise|whole-rise|none"?,"unit":"char|word|line|whole"?,"staggerFrames":n? (offset between unit starts; or staggerSeconds / stagger ticks),"durationFrames":n? (each unit's move),"delayFrames":n? (wait after the clip start),"totalFrames":n? (instead of stagger: the whole entrance ends then),"rise":px? (starts this far below and rises into place; negative drops in),"driftX":px?,"fade":bool|0-100?,"scale":%? (start scale, 100 = none),"easeIn":0-100?,"easeOut":0-100? (Bézier influences),"order":"forward|backward|centerOut|random"?,"seed":n?,"intro":{same fields}|false?,"outro":{durationFrames,staggerFrames,delayFrames (before the clip end),rise (moves up while leaving),fade,scale,easeIn,easeOut,order}|true|false?,"off":bool?} → the stored animation and each unit's start/end (clip and sequence ticks/frames)"#,
            enabled: has_graphic,
            run: animate_text,
            journal: true,
        },
        CommandSpec {
            id: "graphics.textLayout",
            label: "Text Layout",
            menu: &[],
            shortcut: None,
            params: r#"{"clip":id?,"layer":n|name?,"unit":"word|char|line|whole"="word","time":ticks?|"frame"|"seconds"|"timecode" (default: playhead),"animated":bool=false (true: boxes where the animation has them at that time; false: at rest)} → units [{index,text,line,rect:[x,y,w,h] sequence px,center,baselineY,opacity}], lines [{text,rect,width}], block rect, balance {lineWidths,ratio,orphan}, safe {insideFrame,actionSafe,titleSafe}"#,
            enabled: always,
            run: text_layout_cmd,
            journal: false,
        },
        CommandSpec {
            id: "graphics.emphasisTitle",
            label: "Emphasis Title",
            menu: &[],
            shortcut: None,
            params: r##"{"text":str,"placement":"lowerCenter|upperLeft|upperRight|center|leftThird|rightThird"="lowerCenter","size":px=222×height/2160,"font":str? (default Bodoni 72 Bold, else bundled Noto Serif Bold),"fontStyle":str?,"tracking":n=15,"caps":bool=true,"maxLines":n=3,"maxWidth":px?,"color":"#ffffff","animate":"word-rise"|preset|{graphics.animateText params}|false="word-rise","seconds":f64=5 (duration),"time"|"frame"|"timecode":placement?,"track":index?} → clip, balanced lines, font used, position, word layout and animation schedule"##,
            enabled: has_seq,
            run: emphasis_title,
            journal: true,
        },
        CommandSpec {
            id: "graphics.textAnimationPresets",
            label: "Text Animation Presets",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: |_, _| {
                Ok(json!(
                    text_anim::presets()
                        .into_iter()
                        .map(|(id, d, a)| json!({"id": id, "description": d, "animation": serde_json::to_value(&a).unwrap_or_default()}))
                        .collect::<Vec<_>>()
                ))
            },
            journal: false,
        },
    ]
}

#[cfg(test)]
#[path = "text_titles_tests.rs"]
mod tests;
