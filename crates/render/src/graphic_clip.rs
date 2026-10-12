//! Graphic clips: text and shape layers drawn as vectors straight into the output (no resampling),
//! with fill, up to two strokes (outer / centre / inner), a background box and a drop shadow.
//! Each layer is rasterised into a tight premultiplied linear-light image that is cached while
//! the layer and its transform stay the same.
//!
//! [`item_layer_specs`] evaluates a graphic clip's layers the way they are shown: keyframes read
//! through Responsive Design – Time, per-character style runs, responsive pins resolved (pinned
//! layers move, pinned shapes resize), and the roll / crawl offset applied to every layer.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use filmcraft_geom::{Affine, Vec2};
use filmcraft_project::graphic::{Appearance, LayerContent, LayerSpec, LayerTransform, ShapeProps, TextProps, eval_layer};
use filmcraft_project::graphic_design::{GraphicMeta, PinNode, remap_time, resolve_pins, roll_offset};
use filmcraft_project::text_anim::{TextUnit, UnitState, unit_state};
use filmcraft_project::{EffectInstance, TrackItem};
use filmcraft_text::raster::{Xform, apply, fill_into, xform_scale};
use filmcraft_text::{Align, Caps, Layout, Mask, ParagraphStyle, StrokeKind, StyleRun, TextStyle, layout_rich, mask, render};
use filmcraft_time::Tick;
use rayon::prelude::*;

use crate::graphics::premul_linear;
use crate::image::Image;

/// f64 affine → the text crate's f32 transform.
pub fn xform(a: &Affine) -> Xform {
    [a.a as f32, a.b as f32, a.c as f32, a.d as f32, a.e as f32, a.f as f32]
}

/// Layer pixels → graphic canvas pixels (position / anchor / scale / rotation).
pub fn layer_matrix(spec: &LayerSpec) -> Affine {
    let t = &spec.transform;
    Affine::motion(t.position, t.scale, t.rotation, t.anchor)
}

/// Text style + paragraph style of a text layer.
pub fn text_styles(t: &TextProps) -> (TextStyle, ParagraphStyle) {
    (
        TextStyle {
            family: if t.font.is_empty() { filmcraft_text::fonts::DEFAULT_FAMILY.into() } else { t.font.clone() },
            style: if t.style.is_empty() { "Regular".into() } else { t.style.clone() },
            size: t.size,
            tracking: t.tracking,
            kerning: t.kerning,
            ligatures: t.ligatures,
            baseline_shift: t.baseline_shift,
            faux_bold: t.faux_bold,
            faux_italic: t.faux_italic,
            caps: caps_of(t.caps),
            underline: t.underline,
        },
        ParagraphStyle {
            align: match t.align {
                1 => Align::Center,
                2 => Align::Right,
                3 => Align::Justify,
                _ => Align::Left,
            },
            leading: t.leading,
            width: (t.box_width > 0.0 && !t.vertical).then_some(t.box_width),
            height: (t.box_width > 0.0 && t.box_height > 0.0 && !t.vertical).then_some(t.box_height),
            rtl: None,
            vertical: t.vertical,
        },
    )
}

fn caps_of(c: u32) -> Caps {
    match c {
        1 => Caps::All,
        2 => Caps::Small,
        _ => Caps::Normal,
    }
}

/// The per-character style runs of a text layer as text-engine runs (overrides applied to the
/// layer's style).
pub fn text_runs(t: &TextProps, base: &TextStyle) -> Vec<StyleRun> {
    t.runs
        .iter()
        .map(|(r, c)| StyleRun {
            range: r.clone(),
            style: TextStyle {
                family: c.font.clone().filter(|f| !f.is_empty()).unwrap_or_else(|| base.family.clone()),
                style: c.font_style.clone().filter(|f| !f.is_empty()).unwrap_or_else(|| base.style.clone()),
                size: c.size.map_or(base.size, |s| s.max(0.1)),
                tracking: c.tracking.unwrap_or(base.tracking),
                baseline_shift: c.baseline_shift.unwrap_or(base.baseline_shift),
                faux_bold: c.faux_bold.unwrap_or(base.faux_bold),
                faux_italic: c.faux_italic.unwrap_or(base.faux_italic),
                caps: c.caps.map_or(base.caps, caps_of),
                underline: c.underline.unwrap_or(base.underline),
                ..base.clone()
            },
        })
        .collect()
}

/// The laid-out text of a text layer (cached by the text engine).
pub fn text_layout(t: &TextProps) -> Arc<Layout> {
    let (st, ps) = text_styles(t);
    let runs = text_runs(t, &st);
    layout_rich(&t.text, &st, &runs, &ps)
}

/// A shape layer's outline in layer pixels (centred on the layer origin).
pub fn shape_path(s: &ShapeProps) -> filmcraft_text::Path {
    let (w, h) = (s.size.0 / 2.0, s.size.1 / 2.0);
    match s.shape {
        1 => filmcraft_text::Path::ellipse(0.0, 0.0, w, h),
        2 => {
            let n = s.sides.max(3);
            let pts: Vec<(f32, f32)> = (0..n)
                .map(|i| {
                    let a = std::f32::consts::TAU * i as f32 / n as f32 - std::f32::consts::FRAC_PI_2;
                    (a.cos() * w, a.sin() * h)
                })
                .collect();
            filmcraft_text::Path::polygon(&pts)
        }
        3 if s.points.len() >= 3 => filmcraft_text::Path::polygon(&s.points.iter().map(|p| (p[0], p[1])).collect::<Vec<_>>()),
        _ => filmcraft_text::Path::round_rect(-w, -h, w, h, s.corner_radius),
    }
}

/// Content bounds `[x0, y0, x1, y1]` of a layer in layer pixels (text box or shape box).
pub fn layer_local_bounds(spec: &LayerSpec) -> [f32; 4] {
    match &spec.content {
        LayerContent::Text(t) => text_layout(t).bounds,
        LayerContent::Shape(s) => match shape_path(s).bounds() {
            Some(b) => [b.0, b.1, b.2, b.3],
            None => [0.0; 4],
        },
    }
}

/// The four corners of a layer's content box in graphic canvas pixels (selection boxes, hit
/// testing, alignment), clockwise from the top-left.
pub fn layer_quad(spec: &LayerSpec) -> [Vec2; 4] {
    let b = layer_local_bounds(spec);
    let m = layer_matrix(spec);
    [Vec2::new(b[0] as f64, b[1] as f64), Vec2::new(b[2] as f64, b[1] as f64), Vec2::new(b[2] as f64, b[3] as f64), Vec2::new(b[0] as f64, b[3] as f64)]
        .map(|p| m.apply(p))
}

/// A rendered layer: premultiplied linear RGBA at device position (`x`, `y`).
#[derive(Debug)]
pub struct LayerRaster {
    pub x: i32,
    pub y: i32,
    pub img: Image,
}

fn stroke_kind(k: u32) -> StrokeKind {
    match k {
        1 => StrokeKind::Center,
        2 => StrokeKind::Inner,
        _ => StrokeKind::Outer,
    }
}

fn bbox_of(m: &Xform, b: [f32; 4], pad: f32) -> [f32; 4] {
    let pts = [(b[0] - pad, b[1] - pad), (b[2] + pad, b[1] - pad), (b[0] - pad, b[3] + pad), (b[2] + pad, b[3] + pad)];
    let mut o = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for (x, y) in pts {
        let (a, c) = apply(m, x, y);
        o = [o[0].min(a), o[1].min(c), o[2].max(a), o[3].max(c)];
    }
    o
}

fn union(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]
}

/// Over-composite `c` (premultiplied) through `m` into `px` (premultiplied RGBA, same size).
fn over_mask(px: &mut [f32], m: &Mask, c: [f32; 4]) {
    if c[3] <= 0.0 || m.w == 0 {
        return;
    }
    px.par_chunks_mut(4 * m.w).zip(m.a.par_chunks(m.w)).for_each(|(row, mrow)| {
        for (p, &a) in row.as_chunks_mut::<4>().0.iter_mut().zip(mrow) {
            if a > 0.0 {
                let k = 1.0 - c[3] * a;
                for i in 0..4 {
                    p[i] = c[i] * a + p[i] * k;
                }
            }
        }
    });
}

/// Rasterise one layer with layer→device transform `m`, clipped to a `cw`×`ch` canvas. A text
/// layer with a staggered animation ([`LayerSpec::anim`]) draws each word / character / line at
/// its animated offset, opacity and scale.
pub fn raster_layer(spec: &LayerSpec, m: &Xform, cw: usize, ch: usize) -> Option<LayerRaster> {
    if !spec.enabled || spec.transform.opacity <= 0.0 {
        return None;
    }
    match &spec.content {
        LayerContent::Text(t) => {
            let l = text_layout(t);
            if let Some(units) = animated_units(spec, &l) {
                return raster_animated(spec, &l, &units, m, cw, ch);
            }
            raster_with(spec, Some(&l), m, cw, ch)
        }
        LayerContent::Shape(_) => raster_with(spec, None, m, cw, ch),
    }
}

/// [`raster_layer`] with the text already laid out (`text` is None for shapes).
fn raster_with(spec: &LayerSpec, text: Option<&Layout>, m: &Xform, cw: usize, ch: usize) -> Option<LayerRaster> {
    if !spec.enabled || spec.transform.opacity <= 0.0 {
        return None;
    }
    let s = xform_scale(m).max(1e-6);
    let ap = &spec.appearance;
    if let Some(l) = text
        && l.glyphs.is_empty()
        && l.underlines.is_empty()
        && ap.background.is_none()
    {
        return None;
    }
    let lb = match text {
        Some(l) => l.bounds,
        None => layer_local_bounds(spec),
    };
    let shape = match &spec.content {
        LayerContent::Shape(sh) => Some(shape_path(sh).transformed(m)),
        _ => None,
    };
    let stroke_out = ap
        .strokes
        .iter()
        .map(|(_, w, k)| {
            if *k == 2 {
                0.0
            } else if *k == 1 {
                w / 2.0
            } else {
                *w
            }
        })
        .fold(0.0f32, f32::max);
    let overhang = text.map_or(0.0, |l| l.glyphs.iter().map(|g| g.size).fold(0.0f32, f32::max) * 0.35);
    let mut bb = bbox_of(m, lb, overhang + stroke_out + 2.0 / s);
    if let Some((_, pad, _)) = ap.background {
        bb = union(bb, bbox_of(m, lb, pad + 1.0 / s));
    }
    let mut shadow_dev = (0.0f32, 0.0f32);
    if let Some(sh) = &ap.shadow {
        // the offset is in layer space, so it turns and scales with the layer
        shadow_dev = (m[0] * sh.offset.0 + m[2] * sh.offset.1, m[1] * sh.offset.0 + m[3] * sh.offset.1);
        let grow = (sh.size + sh.blur) * s + 2.0;
        bb = union(bb, [bb[0] + shadow_dev.0 - grow, bb[1] + shadow_dev.1 - grow, bb[2] + shadow_dev.0 + grow, bb[3] + shadow_dev.1 + grow]);
    }
    // clip to the canvas, keeping enough margin for blur to read content just outside it
    let margin = ap.shadow.as_ref().map_or(2.0, |sh| (sh.blur + sh.size) * s + shadow_dev.0.abs().max(shadow_dev.1.abs()) + 2.0);
    let x0 = bb[0].floor().max(-margin) as i32;
    let y0 = bb[1].floor().max(-margin) as i32;
    let x1 = bb[2].ceil().min(cw as f32 + margin) as i32;
    let y1 = bb[3].ceil().min(ch as f32 + margin) as i32;
    if x1 <= x0 || y1 <= y0 || x1 <= 0 || y1 <= 0 || x0 >= cw as i32 || y0 >= ch as i32 {
        return None;
    }
    let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
    if w * h > 64 << 20 {
        return None;
    }
    let mut cover = Mask::new(w, h);
    if let Some(l) = text {
        render::draw(l, m, &mut cover, (x0, y0));
    }
    if let Some(p) = &shape {
        fill_into(p, &mut cover, -x0 as f32, -y0 as f32);
    }
    let strokes: Vec<(Mask, [f32; 4])> = ap.strokes.iter().map(|(c, sw, k)| (mask::stroke(&cover, sw * s, stroke_kind(*k)), *c)).collect();
    let bg = ap.background.map(|(c, pad, r)| {
        let p = filmcraft_text::Path::round_rect(lb[0] - pad, lb[1] - pad, lb[2] + pad, lb[3] + pad, r).transformed(m);
        let mut bm = Mask::new(w, h);
        fill_into(&p, &mut bm, -x0 as f32, -y0 as f32);
        (bm, c)
    });
    let mut px = vec![0.0f32; w * h * 4];
    if let Some(sh) = &ap.shadow {
        let mut u = if ap.fill.is_some() { cover.clone() } else { Mask::new(w, h) };
        for (sm, _) in &strokes {
            u.max_with(sm);
        }
        if let Some((bm, _)) = &bg {
            u.max_with(bm);
        }
        if sh.size > 0.0 {
            u = mask::stroke(&u, sh.size * s, StrokeKind::Outer);
        }
        let u = mask::offset(&u, shadow_dev.0.round() as i32, shadow_dev.1.round() as i32);
        let u = mask::blur(&u, sh.blur * s / 3.0);
        over_mask(&mut px, &u, premul_linear(sh.color));
    }
    if let Some((bm, c)) = &bg {
        over_mask(&mut px, bm, premul_linear(*c));
    }
    // outer strokes sit under the fill; centre and inner strokes are drawn over it
    let outer = |k: u32| k != 1 && k != 2;
    for ((sm, c), (_, _, k)) in strokes.iter().zip(&ap.strokes).rev() {
        if outer(*k) {
            over_mask(&mut px, sm, premul_linear(*c));
        }
    }
    if let Some(c) = ap.fill {
        // per-character fills: each style run with its own colour is drawn on its own
        let run_fills: Vec<Option<[f32; 4]>> = match &spec.content {
            LayerContent::Text(t) => t.runs.iter().map(|(_, s)| s.fill).collect(),
            _ => Vec::new(),
        };
        match text {
            Some(l) if run_fills.iter().any(Option::is_some) => {
                let mut groups: Vec<([f32; 4], Vec<u16>)> = Vec::new();
                for r in 0..=run_fills.len() {
                    let col = if r == 0 { c } else { run_fills.get(r - 1).copied().flatten().unwrap_or(c) };
                    match groups.iter_mut().find(|g| g.0 == col) {
                        Some(g) => g.1.push(r as u16),
                        None => groups.push((col, vec![r as u16])),
                    }
                }
                for (col, runs) in groups {
                    let mut m_run = Mask::new(w, h);
                    for r in runs {
                        render::draw_run(l, m, &mut m_run, (x0, y0), Some(r));
                    }
                    over_mask(&mut px, &m_run, premul_linear(col));
                }
            }
            _ => over_mask(&mut px, &cover, premul_linear(c)),
        }
    }
    for ((sm, c), (_, _, k)) in strokes.iter().zip(&ap.strokes).rev() {
        if !outer(*k) {
            over_mask(&mut px, sm, premul_linear(*c));
        }
    }
    let op = spec.transform.opacity;
    if op < 1.0 {
        px.iter_mut().for_each(|v| *v *= op);
    }
    Some(LayerRaster { x: x0, y: y0, img: Image { w, h, px } })
}

/// One animation unit of a laid-out text (a character, word, line or the whole block).
#[derive(Clone, Debug, PartialEq)]
pub struct TextUnitBox {
    /// Byte range of the unit in the layer's text.
    pub range: std::ops::Range<usize>,
    /// Line index (the first line for [`TextUnit::Whole`]).
    pub line: usize,
    /// Logical box `[x0, y0, x1, y1]` in layer pixels: caret extent × the line's ascent / descent.
    pub rect: [f32; 4],
    /// Baseline (layer pixels) of the unit's line.
    pub baseline: f32,
    /// Indices into [`Layout::glyphs`].
    pub glyphs: Vec<usize>,
}

/// Split a laid-out text into units: non-space characters, words (runs of non-space characters
/// within a line), lines (without surrounding spaces; empty lines are skipped) or the whole
/// block. Vertical text has only the whole-block unit.
pub fn text_units(l: &Layout, text: &str, unit: TextUnit) -> Vec<TextUnitBox> {
    let unit = if l.vertical { TextUnit::Whole } else { unit };
    let mut out: Vec<TextUnitBox> = Vec::new();
    for (li, line) in l.lines.iter().enumerate() {
        let Some(lt) = text.get(line.range.clone()) else { continue };
        let base = line.range.start;
        let mut pieces: Vec<std::ops::Range<usize>> = Vec::new();
        match unit {
            TextUnit::Char => {
                for (b, c) in lt.char_indices().filter(|(_, c)| !c.is_whitespace()) {
                    pieces.push(base + b..base + b + c.len_utf8());
                }
            }
            TextUnit::Word => {
                let mut start: Option<usize> = None;
                for (b, c) in lt.char_indices() {
                    match (c.is_whitespace(), start) {
                        (false, None) => start = Some(b),
                        (true, Some(a)) => {
                            pieces.push(base + a..base + b);
                            start = None;
                        }
                        _ => {}
                    }
                }
                if let Some(a) = start {
                    pieces.push(base + a..base + lt.len());
                }
            }
            TextUnit::Line | TextUnit::Whole => {
                let lead = lt.len() - lt.trim_start().len();
                let trimmed = lt.trim();
                if !trimmed.is_empty() {
                    pieces.push(base + lead..base + lead + trimmed.len());
                }
            }
        }
        let (top, bot) = (line.baseline - line.ascent, line.baseline + line.descent);
        for r in pieces {
            let xs = line.carets.iter().filter(|(b, _)| *b >= r.start && *b <= r.end).map(|c| c.1);
            let (mut x0, mut x1) = xs.fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(x), b.max(x)));
            let glyphs: Vec<usize> = (line.glyphs.start..line.glyphs.end.min(l.glyphs.len()))
                .filter(|&g| l.glyphs.get(g).is_some_and(|g| g.cluster >= r.start && g.cluster < r.end))
                .collect();
            if x0 > x1 {
                // no caret stops (should not happen): fall back to the glyph origins
                for g in glyphs.iter().filter_map(|&g| l.glyphs.get(g)) {
                    x0 = x0.min(g.x);
                    x1 = x1.max(g.x + g.size * 0.5);
                }
            }
            if x0 > x1 {
                continue;
            }
            out.push(TextUnitBox { range: r, line: li, rect: [x0, top, x1, bot], baseline: line.baseline, glyphs });
        }
    }
    if unit == TextUnit::Whole && !out.is_empty() {
        let mut all = out.remove(0);
        for u in out.drain(..) {
            all.rect = [all.rect[0].min(u.rect[0]), all.rect[1].min(u.rect[1]), all.rect[2].max(u.rect[2]), all.rect[3].max(u.rect[3])];
            all.range = all.range.start.min(u.range.start)..all.range.end.max(u.range.end);
            all.glyphs.extend(u.glyphs);
        }
        // vertical text and anything the line walk missed: every glyph moves with the block
        all.glyphs = (0..l.glyphs.len()).collect();
        out.push(all);
    }
    out
}

/// The units of an animated text layer with their state now, or None when the layer has no
/// animation or every unit is at rest (then it draws exactly like a still layer).
pub fn animated_units(spec: &LayerSpec, l: &Layout) -> Option<Vec<(TextUnitBox, UnitState)>> {
    let clock = spec.anim.as_ref()?;
    let LayerContent::Text(t) = &spec.content else { return None };
    let units = text_units(l, &t.text, clock.anim.unit);
    let n = units.len();
    let v: Vec<(TextUnitBox, UnitState)> = units.into_iter().enumerate().map(|(i, u)| (u, unit_state(&clock.anim, i, n, clock.time, clock.length))).collect();
    if v.iter().all(|(_, st)| st.is_rest()) { None } else { Some(v) }
}

/// Layer pixels of a unit → layer pixels as animated (scaled about the unit's centre, then
/// offset).
pub fn unit_matrix(rect: [f32; 4], st: &UnitState) -> Affine {
    let (cx, cy) = (((rect[0] + rect[2]) * 0.5) as f64, ((rect[1] + rect[3]) * 0.5) as f64);
    let k = st.scale as f64;
    Affine::translate(cx + st.offset[0] as f64, cy + st.offset[1] as f64).then_apply(&Affine::scale(k, k)).then_apply(&Affine::translate(-cx, -cy))
}

/// Draw an animated text layer: the background (if any) stays put; every visible unit is drawn
/// with its own transform and opacity.
fn raster_animated(spec: &LayerSpec, l: &Layout, units: &[(TextUnitBox, UnitState)], m: &Xform, cw: usize, ch: usize) -> Option<LayerRaster> {
    let mut parts: Vec<LayerRaster> = Vec::new();
    if spec.appearance.background.is_some() {
        let empty = Layout { glyphs: Vec::new(), lines: Vec::new(), underlines: Vec::new(), underline_runs: Vec::new(), ..l.clone() };
        let bg_only = LayerSpec { appearance: Appearance { fill: None, strokes: Vec::new(), ..spec.appearance.clone() }, ..spec.clone() };
        parts.extend(raster_with(&bg_only, Some(&empty), m, cw, ch));
    }
    let unit_spec = LayerSpec { appearance: Appearance { background: None, ..spec.appearance.clone() }, anim: None, ..spec.clone() };
    let lm = Affine { a: m[0] as f64, b: m[1] as f64, c: m[2] as f64, d: m[3] as f64, e: m[4] as f64, f: m[5] as f64 };
    for (u, st) in units {
        if st.alpha <= 0.0 || st.scale <= 1e-6 {
            continue;
        }
        let glyphs: Vec<_> = u.glyphs.iter().filter_map(|&g| l.glyphs.get(g).copied()).collect();
        let mut underlines = Vec::new();
        let mut underline_runs = Vec::new();
        for (k, r) in l.underlines.iter().enumerate() {
            let mid = (r[0] + r[2]) * 0.5;
            if mid >= u.rect[0] && mid <= u.rect[2] && r[1] >= u.rect[1] - 1.0 && r[1] <= u.rect[3] + l.lines.get(u.line).map_or(0.0, |x| x.descent) {
                underlines.push(*r);
                underline_runs.push(l.underline_runs.get(k).copied().unwrap_or(0));
            }
        }
        if glyphs.is_empty() && underlines.is_empty() {
            continue;
        }
        let sub = Layout { glyphs, lines: Vec::new(), underlines, underline_runs, bounds: u.rect, ..l.clone() };
        let us = LayerSpec { transform: LayerTransform { opacity: spec.transform.opacity * st.alpha, ..spec.transform.clone() }, ..unit_spec.clone() };
        let um = xform(&lm.then_apply(&unit_matrix(u.rect, st)));
        parts.extend(raster_with(&us, Some(&sub), &um, cw, ch));
    }
    let x0 = parts.iter().map(|r| r.x).min()?;
    let y0 = parts.iter().map(|r| r.y).min()?;
    let x1 = parts.iter().map(|r| r.x.saturating_add(r.img.w as i32)).max()?;
    let y1 = parts.iter().map(|r| r.y.saturating_add(r.img.h as i32)).max()?;
    let (w, h) = ((x1 - x0).max(1) as usize, (y1 - y0).max(1) as usize);
    if w.saturating_mul(h) > 64 << 20 {
        return None;
    }
    let mut img = Image::new(w, h);
    for r in &parts {
        composite_raster(&mut img, r, -x0, -y0);
    }
    Some(LayerRaster { x: x0, y: y0, img })
}

type Cache = Mutex<HashMap<u64, (u64, Option<Arc<LayerRaster>>)>>;

fn raster_cache() -> &'static Cache {
    static C: OnceLock<Cache> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// [`raster_layer`] through a small cache (static titles re-composite without re-rasterising).
pub fn raster_layer_cached(spec: &LayerSpec, m: &Xform, cw: usize, ch: usize) -> Option<Arc<LayerRaster>> {
    let mut hs = std::collections::hash_map::DefaultHasher::new();
    // an animated text layer is keyed by what its units look like, not by the clock: a held
    // title re-composites from the cache, and a finished animation is the plain layer
    let settled;
    let spec = match (&spec.anim, &spec.content) {
        (Some(_), LayerContent::Text(t)) => {
            let l = text_layout(t);
            match animated_units(spec, &l) {
                Some(units) => {
                    if spec.appearance.background.is_none() && units.iter().all(|(_, st)| st.alpha <= 0.0) {
                        return None;
                    }
                    units.iter().for_each(|(u, st)| format!("{:?}{:?}", u.glyphs, st).hash(&mut hs));
                    format!("{:?}", LayerSpec { anim: None, ..spec.clone() }).hash(&mut hs);
                    spec
                }
                None => {
                    settled = LayerSpec { anim: None, ..spec.clone() };
                    &settled
                }
            }
        }
        _ => spec,
    };
    format!("{spec:?}").hash(&mut hs);
    m.iter().for_each(|v| v.to_bits().hash(&mut hs));
    (cw, ch).hash(&mut hs);
    let key = hs.finish();
    static CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = CLOCK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Some(e) = raster_cache().lock().unwrap_or_else(|e| e.into_inner()).get_mut(&key) {
        e.0 = now;
        return e.1.clone();
    }
    let r = raster_layer(spec, m, cw, ch).map(Arc::new);
    let mut c = raster_cache().lock().unwrap_or_else(|e| e.into_inner());
    if c.len() >= 96
        && let Some(k) = c.iter().min_by_key(|(_, v)| v.0).map(|(k, _)| *k)
    {
        c.remove(&k);
    }
    c.insert(key, (now, r.clone()));
    r
}

/// Over-composite a raster at (`r.x` + `dx`, `r.y` + `dy`) onto `img`.
pub fn composite_raster(img: &mut Image, r: &LayerRaster, dx: i32, dy: i32) {
    let (rx, ry) = (r.x + dx, r.y + dy);
    let (w, h) = (img.w as i32, img.h as i32);
    let ys = ry.max(0)..(ry + r.img.h as i32).min(h);
    let xs = rx.max(0)..(rx + r.img.w as i32).min(w);
    if ys.is_empty() || xs.is_empty() {
        return;
    }
    let iw = img.w;
    img.px[ys.start as usize * iw * 4..ys.end as usize * iw * 4].par_chunks_mut(iw * 4).enumerate().for_each(|(row_i, row)| {
        let sy = (ys.start + row_i as i32 - ry) as usize;
        for x in xs.clone() {
            let si = (sy * r.img.w + (x - rx) as usize) * 4;
            let s = &r.img.px[si..si + 4];
            if s[3] <= 0.0 {
                continue;
            }
            let d = &mut row[x as usize * 4..x as usize * 4 + 4];
            let k = 1.0 - s[3];
            for c in 0..4 {
                d[c] = s[c] + d[c] * k;
            }
        }
    });
}

/// Axis-aligned bounds `[x0, y0, x1, y1]` of a layer's content box in canvas pixels.
pub fn layer_bounds(spec: &LayerSpec) -> [f64; 4] {
    let q = layer_quad(spec);
    let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for p in q {
        b = [b[0].min(p.x), b[1].min(p.y), b[2].max(p.x), b[3].max(p.y)];
    }
    b
}

/// Resolve responsive pins of evaluated layers in place: pinned layers move, shapes pinned on
/// opposite edges resize.
pub fn apply_pins(specs: &mut [LayerSpec], canvas: (u32, u32)) {
    if !specs.iter().any(|s| s.pin.is_some()) {
        return;
    }
    let nodes: Vec<PinNode> = specs
        .iter()
        .map(|s| PinNode { uid: s.uid, pin: s.pin.clone(), bounds: layer_bounds(s), resizable: matches!(s.content, LayerContent::Shape(_)) })
        .collect();
    let out = resolve_pins(&nodes, [0.0, 0.0, canvas.0 as f64, canvas.1 as f64]);
    for ((s, n), b) in specs.iter_mut().zip(&nodes).zip(out) {
        if b == n.bounds {
            continue;
        }
        let (ow, oh) = (n.bounds[2] - n.bounds[0], n.bounds[3] - n.bounds[1]);
        let (nw, nh) = (b[2] - b[0], b[3] - b[1]);
        if let LayerContent::Shape(sh) = &mut s.content {
            let kx = if ow > 1e-9 { nw / ow } else { 1.0 };
            let ky = if oh > 1e-9 { nh / oh } else { 1.0 };
            if (kx - 1.0).abs() > 1e-12 || (ky - 1.0).abs() > 1e-12 {
                sh.size = ((sh.size.0 as f64 * kx) as f32, (sh.size.1 as f64 * ky) as f32);
                sh.points.iter_mut().for_each(|p| *p = [(p[0] as f64 * kx) as f32, (p[1] as f64 * ky) as f32]);
            }
        }
        let now = layer_bounds(s);
        s.transform.position = Vec2::new(s.transform.position.x + b[0] - now[0], s.transform.position.y + b[1] - now[1]);
    }
}

/// Evaluated layers of a graphic clip's effect list at clip time `t`, in paint order, with pins
/// resolved.
pub fn graphic_specs(effects: &[EffectInstance], t: Tick, canvas: (u32, u32)) -> Vec<LayerSpec> {
    let mut v: Vec<LayerSpec> = effects.iter().filter(|e| filmcraft_project::graphic::is_layer(e)).filter_map(|e| eval_layer(e, t, canvas)).collect();
    apply_pins(&mut v, canvas);
    v
}

/// The media time at which a graphic clip's layers are read at media time `mt` (Responsive Design
/// – Time), and the clip-relative time and clip length (media time) for rolls.
pub fn graphic_time(item: &TrackItem, mt: Tick) -> (Tick, Tick, Tick) {
    let u = mt - item.source_in;
    let d = Tick((item.duration.0 as f64 * item.speed.abs().max(1e-9)).round() as i64);
    let t = match item.graphic.as_deref() {
        Some(m) if m.has_responsive_time() => remap_time(m, u, d),
        _ => mt,
    };
    (t, u, d)
}

/// Union of the layers' bounds (with background padding) for rolls and crawls.
fn content_bounds(specs: &[LayerSpec]) -> Option<[f64; 4]> {
    let mut b: Option<[f64; 4]> = None;
    for s in specs.iter().filter(|s| s.enabled) {
        let mut lb = layer_bounds(s);
        if let Some((_, pad, _)) = s.appearance.background {
            let k = s.transform.scale.x.abs().max(s.transform.scale.y.abs());
            let p = pad as f64 * k;
            lb = [lb[0] - p, lb[1] - p, lb[2] + p, lb[3] + p];
        }
        b = Some(match b {
            Some(a) => [a[0].min(lb[0]), a[1].min(lb[1]), a[2].max(lb[2]), a[3].max(lb[3])],
            None => lb,
        });
    }
    b
}

/// A graphic clip's layers at media time `mt` as shown: `(effect index, layer)` in paint order,
/// with Responsive Design – Time, pins and the roll / crawl offset applied. Includes hidden
/// layers (their `enabled` is false).
pub fn item_layer_specs(item: &TrackItem, mt: Tick, canvas: (u32, u32)) -> Vec<(usize, LayerSpec)> {
    let (t, u, d) = graphic_time(item, mt);
    let mut idx = Vec::new();
    let mut specs = Vec::new();
    for (i, e) in item.effects.iter().enumerate() {
        if filmcraft_project::graphic::is_layer(e)
            && let Some(s) = eval_layer(e, t, canvas)
        {
            idx.push(i);
            specs.push(s);
        }
    }
    apply_pins(&mut specs, canvas);
    if let Some(meta) = item.graphic.as_deref() {
        apply_roll(meta, &mut specs, u, d, canvas);
    }
    // staggered text animations run on clip time: the entrance from the clip start, the exit
    // up to the clip end (unaffected by Responsive Design – Time remapping)
    for s in specs.iter_mut() {
        if let Some(c) = s.anim.as_mut() {
            c.time = u;
            c.length = Some(d);
        }
    }
    idx.into_iter().zip(specs).collect()
}

/// Move every layer by the roll / crawl offset at clip time `u` of a clip `d` long.
pub fn apply_roll(meta: &GraphicMeta, specs: &mut [LayerSpec], u: Tick, d: Tick, canvas: (u32, u32)) {
    if meta.roll.mode == filmcraft_project::RollMode::Off {
        return;
    }
    let Some(cb) = content_bounds(specs) else { return };
    let (dx, dy) = roll_offset(&meta.roll, u, d, canvas, cb);
    for s in specs.iter_mut() {
        s.transform.position = Vec2::new(s.transform.position.x + dx, s.transform.position.y + dy);
    }
}

/// Draw all layers of a graphic clip at media time `mt` onto `out`; `m` maps graphic canvas
/// pixels to `out` pixels.
pub fn render_graphic(item: &TrackItem, mt: Tick, canvas: (u32, u32), m: &Affine, out: &mut Image) {
    let specs: Vec<LayerSpec> = item_layer_specs(item, mt, canvas).into_iter().map(|(_, s)| s).collect();
    draw_specs(&specs, m, out);
}

/// Draw a graphic's layers (an effect list, without clip settings) at time `t` onto `out`.
pub fn render_graphic_layers(effects: &[EffectInstance], t: Tick, canvas: (u32, u32), m: &Affine, out: &mut Image) {
    draw_specs(&graphic_specs(effects, t, canvas), m, out);
}

fn draw_specs(specs: &[LayerSpec], m: &Affine, out: &mut Image) {
    for spec in specs {
        let lm = xform(&m.then_apply(&layer_matrix(spec)));
        if let Some(r) = raster_layer_cached(spec, &lm, out.w, out.h) {
            composite_raster(out, &r, 0, 0);
        }
    }
}

/// All layers composed into one tight image (for the GPU plan): `(image, x, y)` in output pixels.
pub fn render_graphic_tight(item: &TrackItem, mt: Tick, canvas: (u32, u32), m: &Affine, w: usize, h: usize) -> Option<(Image, i32, i32)> {
    let rasters: Vec<Arc<LayerRaster>> =
        item_layer_specs(item, mt, canvas).iter().filter_map(|(_, spec)| raster_layer_cached(spec, &xform(&m.then_apply(&layer_matrix(spec))), w, h)).collect();
    let x0 = rasters.iter().map(|r| r.x).min()?.max(0);
    let y0 = rasters.iter().map(|r| r.y).min()?.max(0);
    let x1 = rasters.iter().map(|r| r.x + r.img.w as i32).max()?.min(w as i32);
    let y1 = rasters.iter().map(|r| r.y + r.img.h as i32).max()?.min(h as i32);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let mut img = Image::new((x1 - x0) as usize, (y1 - y0) as usize);
    for r in &rasters {
        composite_raster(&mut img, r, -x0, -y0);
    }
    Some((img, x0, y0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_project::graphic::{new_shape_layer, new_text_layer};
    use filmcraft_project::{ParamValue, find_effect};

    fn set(e: &mut EffectInstance, k: &str, v: ParamValue) {
        e.params.get_mut(k).unwrap_or_else(|| panic!("{k}")).value = v;
    }

    /// A 5 s graphic clip holding `layers` (canvas 400×300 at 25 fps).
    fn graphic_item(layers: Vec<EffectInstance>) -> TrackItem {
        let mut p = filmcraft_project::Project::new("t");
        let rate = filmcraft_time::FrameRate::FPS_25;
        let g = p.add_item("G", filmcraft_project::Label::Rose, filmcraft_project::ItemKind::Graphic { width: 400, height: 300, rate }, None);
        let mut it = p
            .make_track_item(g, filmcraft_project::TrackKind::Video, Tick::ZERO, filmcraft_time::TimeRange::new(Tick::ZERO, Tick::from_seconds_f64(5.0)), rate)
            .unwrap();
        it.effects.extend(layers);
        it
    }

    fn ink_box(img: &Image) -> Option<[usize; 4]> {
        let mut b: Option<[usize; 4]> = None;
        for y in 0..img.h {
            for x in 0..img.w {
                if img.get(x, y)[3] > 0.5 {
                    b = Some(match b {
                        Some(a) => [a[0].min(x), a[1].min(y), a[2].max(x), a[3].max(y)],
                        None => [x, y, x, y],
                    });
                }
            }
        }
        b
    }

    #[test]
    fn per_character_fill_and_size() {
        use filmcraft_project::graphic_design::{CharStyle, LayerExtra, StyleRun};
        let mut e = new_text_layer("AAAA", Vec2::new(20.0, 100.0), 60.0);
        set(&mut e, "fill_color", ParamValue::Color([1.0, 1.0, 1.0, 1.0]));
        let red = CharStyle { fill: Some([1.0, 0.0, 0.0, 1.0]), ..Default::default() };
        e.layer = Some(Box::new(LayerExtra { runs: vec![StyleRun { start: 2, end: 4, style: red }], ..Default::default() }));
        let mut img = Image::new(400, 200);
        render_graphic_layers(&[e.clone()], Tick::ZERO, (400, 200), &Affine::IDENTITY, &mut img);
        let l = text_layout(match &eval_layer(&e, Tick::ZERO, (400, 200)).unwrap().content {
            LayerContent::Text(t) => t,
            _ => unreachable!(),
        });
        let split = 20.0 + l.caret(2).0;
        let (mut white, mut red_px) = (0, 0);
        for y in 0..200 {
            for x in 0..400 {
                let p = img.get(x, y);
                if p[3] > 0.9 {
                    if p[1] > 0.9 {
                        white += 1;
                        assert!((x as f32) < split + 1.0, "white only in the first half at {x}");
                    } else if p[0] > 0.9 && p[1] < 0.05 {
                        red_px += 1;
                        assert!((x as f32) > split - 1.0, "red only in the second half at {x}");
                    }
                }
            }
        }
        assert!(white > 200 && red_px > 200, "{white} {red_px}");
        // a bigger run is taller and sits on the same baseline
        let big = CharStyle { size: Some(120.0), ..Default::default() };
        e.layer = Some(Box::new(LayerExtra { runs: vec![StyleRun { start: 3, end: 4, style: big }], ..Default::default() }));
        let mut img2 = Image::new(400, 200);
        render_graphic_layers(&[e], Tick::ZERO, (400, 200), &Affine::IDENTITY, &mut img2);
        let (b1, b2) = (ink_box(&img).unwrap(), ink_box(&img2).unwrap());
        assert!(b2[1] + 30 < b1[1], "the big letter rises higher: {b1:?} {b2:?}");
        assert!(b2[3].abs_diff(b1[3]) <= 1, "same baseline: {b1:?} {b2:?}");
    }

    #[test]
    fn pinned_box_follows_text_growth() {
        use filmcraft_project::graphic_design::{LayerExtra, Pin, PinTarget};
        let mut t = new_text_layer("Hi", Vec2::new(50.0, 100.0), 40.0);
        t.layer = Some(Box::new(LayerExtra { uid: 1, ..Default::default() }));
        let mut bx = new_shape_layer(0, Vec2::new(60.0, 90.0), Vec2::new(40.0, 40.0), vec![]);
        let tb = layer_bounds(&eval_layer(&t, Tick::ZERO, (400, 300)).unwrap());
        let padded = [tb[0] - 10.0, tb[1] - 10.0, tb[2] + 10.0, tb[3] + 10.0];
        let pin = Pin::new(PinTarget::Layer(1), [true; 4], padded, tb);
        bx.layer = Some(Box::new(LayerExtra { uid: 2, pin: Some(pin), ..Default::default() }));
        for text in ["Hi", "Hello there, world"] {
            set(&mut t, "text", ParamValue::Text(text.into()));
            let specs = graphic_specs(&[bx.clone(), t.clone()], Tick::ZERO, (400, 300));
            let (b, tt) = (layer_bounds(&specs[0]), layer_bounds(&specs[1]));
            for k in 0..4 {
                let want = tt[k] + if k < 2 { -10.0 } else { 10.0 };
                assert!((b[k] - want).abs() < 1e-3, "{text}: edge {k}: {b:?} vs {tt:?}");
            }
        }
    }

    #[test]
    fn roll_moves_the_whole_graphic() {
        use filmcraft_project::graphic_design::{GraphicMeta, Roll, RollMode};
        let e = new_shape_layer(0, Vec2::new(200.0, 150.0), Vec2::new(100.0, 100.0), vec![]);
        let mut item = graphic_item(vec![e]);
        item.graphic = Some(Box::new(GraphicMeta {
            roll: Roll { mode: RollMode::Roll, start_off_screen: true, end_off_screen: true, ..Default::default() },
            ..Default::default()
        }));
        let at = |item: &TrackItem, s: f64| {
            let sp = item_layer_specs(item, Tick::from_seconds_f64(s), (400, 300));
            layer_bounds(&sp[0].1)
        };
        // the box (y 100..200) starts with its top at the frame bottom and ends above the frame
        assert!((at(&item, 0.0)[1] - 300.0).abs() < 1e-6);
        assert!(at(&item, 5.0)[3].abs() < 1e-6);
        // half way: offset (200 + (-200)) / 2 = 0 → back where it was laid out
        assert!((at(&item, 2.5)[1] - 100.0).abs() < 1e-6, "{:?}", at(&item, 2.5));
        let mut img = Image::new(400, 300);
        render_graphic(&item, Tick::from_seconds_f64(2.5), (400, 300), &Affine::IDENTITY, &mut img);
        assert!(img.get(200, 150)[3] > 0.9, "centre at mid-roll");
        let mut img0 = Image::new(400, 300);
        render_graphic(&item, Tick::ZERO, (400, 300), &Affine::IDENTITY, &mut img0);
        assert!(ink_box(&img0).is_none(), "off screen at the start");
    }

    #[test]
    fn text_layer_renders_white_text_at_position() {
        let e = new_text_layer("HELLO", Vec2::new(100.0, 150.0), 60.0);
        let mut img = Image::new(400, 200);
        render_graphic_layers(&[e], Tick::ZERO, (400, 200), &Affine::IDENTITY, &mut img);
        let ink: Vec<(usize, usize)> = (0..200).flat_map(|y| (0..400).map(move |x| (x, y))).filter(|&(x, y)| img.get(x, y)[3] > 0.5).collect();
        assert!(ink.len() > 500);
        let minx = ink.iter().map(|p| p.0).min().unwrap();
        let maxy = ink.iter().map(|p| p.1).max().unwrap();
        assert!((98..=106).contains(&minx), "starts at the position: {minx}");
        assert!((145..=152).contains(&maxy), "sits on the baseline: {maxy}");
        assert!(img.get(ink[0].0, ink[0].1)[0] > 0.4, "white");
    }

    #[test]
    fn shape_appearance_layers() {
        let mut e = new_shape_layer(0, Vec2::new(100.0, 100.0), Vec2::new(80.0, 40.0), vec![]);
        set(&mut e, "fill_color", ParamValue::Color([0.0, 0.0, 1.0, 1.0]));
        set(&mut e, "stroke", ParamValue::Bool(true));
        set(&mut e, "stroke_width", ParamValue::Float(6.0));
        set(&mut e, "stroke_color", ParamValue::Color([1.0, 0.0, 0.0, 1.0]));
        set(&mut e, "shadow", ParamValue::Bool(true));
        set(&mut e, "shadow_distance", ParamValue::Float(20.0));
        set(&mut e, "shadow_blur", ParamValue::Float(0.0));
        let mut img = Image::new(200, 200);
        render_graphic_layers(&[e], Tick::ZERO, (200, 200), &Affine::IDENTITY, &mut img);
        let p = img.get(100, 100);
        assert!(p[2] > 0.9 && p[0] < 0.05, "blue fill {p:?}");
        let s = img.get(100, 78);
        assert!(s[0] > 0.9 && s[2] < 0.05, "red outer stroke above the box {s:?}");
        // shadow: below-right of the box, outside the stroke
        let sh = img.get(150, 135);
        assert!(sh[3] > 0.5 && sh[0] < 0.05 && sh[2] < 0.05, "black shadow {sh:?}");
        assert_eq!(img.get(20, 20)[3], 0.0);
    }

    #[test]
    fn inner_and_centre_strokes() {
        for (kind, inside, outside) in [(2u32, true, false), (1, true, true)] {
            let mut e = new_shape_layer(2, Vec2::new(100.0, 100.0), Vec2::new(120.0, 120.0), vec![]);
            set(&mut e, "sides", ParamValue::Float(4.0));
            set(&mut e, "stroke", ParamValue::Bool(true));
            set(&mut e, "stroke_width", ParamValue::Float(6.0));
            set(&mut e, "stroke_type", ParamValue::Choice(kind));
            set(&mut e, "stroke_color", ParamValue::Color([0.0, 1.0, 0.0, 1.0]));
            let mut img = Image::new(200, 200);
            render_graphic_layers(&[e], Tick::ZERO, (200, 200), &Affine::IDENTITY, &mut img);
            // diamond: top vertex at y = 40; probe straight below/above the right vertex (160, 100)
            let green = |x: usize| {
                let p = img.get(x, 100);
                p[1] > 0.9 && p[0] < 0.1
            };
            assert_eq!(green(157), inside, "kind {kind} inside");
            assert_eq!(green(161), outside, "kind {kind} outside");
            assert!(!green(120), "fill in the middle");
        }
    }

    #[test]
    fn rotation_scale_and_opacity() {
        let mut e = new_shape_layer(0, Vec2::new(100.0, 100.0), Vec2::new(100.0, 10.0), vec![]);
        set(&mut e, "rotation", ParamValue::Float(90.0));
        set(&mut e, "opacity", ParamValue::Float(50.0));
        let mut img = Image::new(200, 200);
        render_graphic_layers(&[e.clone()], Tick::ZERO, (200, 200), &Affine::IDENTITY, &mut img);
        assert!((img.get(100, 60)[3] - 0.5).abs() < 0.02, "vertical after 90°");
        assert_eq!(img.get(60, 100)[3], 0.0);
        // a 2× output transform (e.g. Motion scale) draws vectors crisply at the larger size
        let mut big = Image::new(400, 400);
        render_graphic_layers(&[e], Tick::ZERO, (200, 200), &Affine::scale(2.0, 2.0), &mut big);
        assert!((big.get(200, 110)[3] - 0.5).abs() < 0.02);
        let q = layer_quad(&eval_layer(&find_effect("graphic_shape").unwrap().instance(), Tick::ZERO, (200, 200)).unwrap());
        assert!((q[0].x - (100.0 - 200.0)).abs() < 1e-6 && (q[2].y - (100.0 + 100.0)).abs() < 1e-6);
    }

    /// "NO EXPERIENCE REQUIRED" with the word-rise preset in a 5 s clip on a 1200×300 canvas.
    fn word_rise_item(unit: Option<filmcraft_project::text_anim::TextUnit>) -> TrackItem {
        use filmcraft_project::graphic_design::LayerExtra;
        let mut e = new_text_layer("NO EXPERIENCE REQUIRED", Vec2::new(40.0, 200.0), 80.0);
        let mut anim = filmcraft_project::text_anim::preset("word-rise").unwrap();
        if let Some(u) = unit {
            anim.unit = u;
        }
        e.layer = Some(Box::new(LayerExtra { anim: Some(anim), ..Default::default() }));
        graphic_item(vec![e])
    }

    fn render_at(item: &TrackItem, t: Tick) -> Image {
        let mut img = Image::new(1200, 300);
        render_graphic(item, t, (1200, 300), &Affine::IDENTITY, &mut img);
        img
    }

    /// Ink (alpha > `thr`) bounding box within columns `x0..x1`.
    fn ink_in(img: &Image, x0: f32, x1: f32, thr: f32) -> Option<[usize; 4]> {
        let mut b: Option<[usize; 4]> = None;
        for y in 0..img.h {
            for x in (x0.max(0.0) as usize)..(x1.min(img.w as f32) as usize) {
                if img.get(x, y)[3] > thr {
                    b = Some(match b {
                        Some(a) => [a[0].min(x), a[1].min(y), a[2].max(x), a[3].max(y)],
                        None => [x, y, x, y],
                    });
                }
            }
        }
        b
    }

    fn max_alpha(img: &Image, x0: f32, x1: f32) -> f32 {
        let mut m = 0.0f32;
        for y in 0..img.h {
            for x in (x0.max(0.0) as usize)..(x1.min(img.w as f32) as usize) {
                m = m.max(img.get(x, y)[3]);
            }
        }
        m
    }

    #[test]
    fn words_enter_one_by_one_and_rise_into_place() {
        use filmcraft_project::text_anim::frames60;
        let item = word_rise_item(None);
        let spec = item_layer_specs(&item, Tick::ZERO, (1200, 300)).remove(0).1;
        let LayerContent::Text(t) = &spec.content else { unreachable!() };
        let l = text_layout(t);
        let units = text_units(&l, &t.text, TextUnit::Word);
        assert_eq!(units.iter().map(|u| &t.text[u.range.clone()]).collect::<Vec<_>>(), vec!["NO", "EXPERIENCE", "REQUIRED"]);
        // word columns in canvas pixels (the layer origin is at x = 40)
        let col = |i: usize| (40.0 + units[i].rect[0] - 2.0, 40.0 + units[i].rect[2] + 2.0);
        let rest = render_at(&item, frames60(120));
        let still = {
            let mut img = Image::new(1200, 300);
            let plain = new_text_layer("NO EXPERIENCE REQUIRED", Vec2::new(40.0, 200.0), 80.0);
            render_graphic(&graphic_item(vec![plain]), Tick::ZERO, (1200, 300), &Affine::IDENTITY, &mut img);
            img
        };
        assert!(rest.px.iter().zip(&still.px).all(|(a, b)| (a - b).abs() < 1e-5), "the hold looks exactly like the still title");
        // frame 0: nothing yet
        assert!(ink_box(&render_at(&item, Tick::ZERO)).is_none());
        // frame 8: only NO, partly faded and still below its resting place
        let f8 = render_at(&item, frames60(8));
        let (a, b) = col(0);
        let no = ink_in(&f8, a, b, 0.05).expect("NO is entering");
        let no_rest = ink_in(&rest, a, b, 0.05).unwrap();
        assert!(no[3] > no_rest[3] + 3, "NO is still below its place: {no:?} vs {no_rest:?}");
        assert!(max_alpha(&f8, a, b) < 0.95, "and not yet opaque");
        for i in [1, 2] {
            let (a, b) = col(i);
            assert_eq!(max_alpha(&f8, a, b), 0.0, "word {i} has not started");
        }
        // frame 24: NO at rest, EXPERIENCE entering, REQUIRED not started
        let f24 = render_at(&item, frames60(24));
        let (a, b) = col(0);
        assert_eq!(ink_in(&f24, a, b, 0.5), ink_in(&rest, a, b, 0.5));
        let (a, b) = col(1);
        assert!(max_alpha(&f24, a, b) > 0.05);
        let (a, b) = col(2);
        assert_eq!(max_alpha(&f24, a, b), 0.0);
        // frame 48: the phrase has landed — on one line, never re-centred while it builds
        let f48 = render_at(&item, frames60(48));
        assert!(f48.px.iter().zip(&rest.px).all(|(a, b)| (a - b).abs() < 1e-5));
        // exit: the last 18 frames fade every word together, nothing moves
        let out = render_at(&item, Tick::from_seconds_f64(5.0) - frames60(9));
        for i in 0..3 {
            let (a, b) = col(i);
            let m = max_alpha(&out, a, b);
            assert!(m > 0.2 && m < 0.8, "word {i} half faded: {m}");
            assert_eq!(ink_in(&out, a, b, 0.01).map(|x| x[3]), ink_in(&rest, a, b, 0.01).map(|x| x[3]), "no exit motion");
        }
        // deterministic: the same frame renders the same pixels
        assert_eq!(render_at(&item, frames60(30)).px, render_at(&item, frames60(30)).px);
    }

    #[test]
    fn characters_and_lines_are_units_too() {
        use filmcraft_project::text_anim::frames60;
        let item = word_rise_item(Some(TextUnit::Char));
        let spec = item_layer_specs(&item, Tick::ZERO, (1200, 300)).remove(0).1;
        let LayerContent::Text(t) = &spec.content else { unreachable!() };
        let l = text_layout(t);
        assert_eq!(text_units(&l, &t.text, TextUnit::Char).len(), 20, "letters, spaces skipped");
        // char mode: 14-frame stagger → at frame 15 only "N" and "O" have started
        let img = render_at(&item, frames60(15));
        let units = text_units(&l, &t.text, TextUnit::Char);
        let started: Vec<bool> = units.iter().map(|u| max_alpha(&img, 40.0 + u.rect[0] + 1.0, 40.0 + u.rect[2] - 1.0) > 0.0).collect();
        assert_eq!(&started[..3], &[true, true, false]);
        // two lines
        let lt = TextProps { text: "ONE TWO\nTHREE".into(), ..t.clone() };
        let ll = text_layout(&lt);
        let lines = text_units(&ll, &lt.text, TextUnit::Line);
        assert_eq!(lines.len(), 2);
        assert!(lines[1].rect[1] > lines[0].rect[1]);
        assert_eq!(text_units(&ll, &lt.text, TextUnit::Whole).len(), 1);
        assert_eq!(text_units(&ll, &lt.text, TextUnit::Word).iter().map(|u| u.line).collect::<Vec<_>>(), vec![0, 0, 1]);
    }

    #[test]
    fn animated_background_box_stays_put() {
        use filmcraft_project::text_anim::frames60;
        let mut item = word_rise_item(None);
        if let Some(e) = item.effects.iter_mut().find(|e| e.effect == "graphic_text") {
            set(e, "background", ParamValue::Bool(true));
            set(e, "background_opacity", ParamValue::Float(100.0));
        }
        // before any word arrives the box is drawn, the words are not
        let img = render_at(&item, Tick::ZERO);
        let b = ink_box(&img).expect("the box shows");
        let rest = ink_box(&render_at(&item, frames60(100))).unwrap();
        assert_eq!(b, rest, "same box as at rest");
    }

    #[test]
    fn tight_image_matches_full_render() {
        let a = new_text_layer("Tight", Vec2::new(50.0, 80.0), 40.0);
        let b = new_shape_layer(1, Vec2::new(150.0, 120.0), Vec2::new(60.0, 60.0), vec![]);
        let fx = vec![b, a];
        let mut full = Image::new(220, 160);
        let item = graphic_item(fx);
        render_graphic(&item, Tick::ZERO, (220, 160), &Affine::IDENTITY, &mut full);
        let (tight, x, y) = render_graphic_tight(&item, Tick::ZERO, (220, 160), &Affine::IDENTITY, 220, 160).unwrap();
        assert!(tight.w < 220 && tight.h < 160);
        for yy in 0..tight.h {
            for xx in 0..tight.w {
                let (p, q) = (tight.get(xx, yy), full.get(xx + x as usize, yy + y as usize));
                assert!((p[3] - q[3]).abs() < 1e-5);
            }
        }
    }
}
