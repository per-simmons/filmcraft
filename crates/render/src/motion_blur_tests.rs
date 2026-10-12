//! Transform motion blur: with "Use Composition's Shutter Angle" off and a Shutter Angle above 0,
//! a moving Transform averages sub-frame samples across the shutter interval, so moving edges
//! smear by the distance they travel while the shutter is open. Static Transforms, a closed
//! shutter and the composition shutter (0° in a sequence, as in Premiere) render exactly as
//! without motion blur.

use super::*;
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{Generator, MediaSource};
use filmcraft_project::{ClipId, Label, MediaClip, MediaRef, SequenceSettings, TrackKind, find_effect};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, TimeRange};

const R: FrameRate = FrameRate::FPS_24;
const W: usize = 320;
const H: usize = 180;

struct Scene {
    p: Project,
    seq: ItemId,
    map: SourceMap,
}

/// A white box (a matte at 30 % Motion scale: 96 × 54 px, centred) on black with an adjustment
/// layer above it (`adjust`), or a full-frame white matte alone (its own Transform makes the
/// box); returns the clip that gets the Transform.
fn scene(adjust: bool) -> (Scene, ClipId) {
    let mut p = Project::new("mb");
    let seq = p.new_sequence("s", SequenceSettings { width: W as u32, height: H as u32, frame_rate: R, ..Default::default() }, 2, 0, None);
    let mut map = SourceMap::default();
    let g = GeneratorSource::new(Generator::ColorMatte { color: [1.0, 1.0, 1.0, 1.0] }, W as u32, H as u32, R, Tick(10 * TICKS_PER_SECOND));
    let info = g.info().clone();
    let media = p.add_item(
        "white",
        Label::Iris,
        ItemKind::Media(MediaClip {
            media: MediaRef::Generator(g.generator.clone()),
            info,
            interpret: Default::default(),
            mark_in: None,
            mark_out: None,
            markers: vec![],
            offline: false,
            proxy: None,
            identity: None,
        }),
        None,
    );
    map.0.insert(media, Arc::new(g) as SharedSource);
    let mut s = Scene { p, seq, map };
    let base = place(&mut s, 0, media);
    if !adjust {
        return (s, base);
    }
    let motion = clip(&mut s, base).effects.iter_mut().find(|e| e.effect == "motion").expect("motion");
    motion.params.get_mut("scale").expect("scale").value = ParamValue::Float(30.0);
    let a = s.p.add_item(
        "Adjustment Layer",
        Label::Lavender,
        ItemKind::AdjustmentLayer { width: W as u32, height: H as u32, rate: R, duration: R.tick_of(48) },
        None,
    );
    let adj = place(&mut s, 1, a);
    (s, adj)
}

fn place(s: &mut Scene, track: usize, item: ItemId) -> ClipId {
    let mut ti = s.p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, R.tick_of(48)), R).expect("item");
    ti.scale_to_frame = true;
    let id = ti.id;
    let q = s.p.sequence_mut(s.seq).expect("seq");
    q.video_tracks[track].items.push(ti);
    q.video_tracks[track].sort();
    id
}

fn clip(s: &mut Scene, c: ClipId) -> &mut TrackItem {
    s.p.sequence_mut(s.seq).expect("seq").find_item_mut(c).expect("clip").1
}

/// Adds a Transform whose Scale goes `from` → `to` between frames `k0` and `k1` (linear), with the
/// given composition switch and shutter angle.
fn push(s: &mut Scene, c: ClipId, from: f64, to: f64, k0: i64, k1: i64, comp: bool, angle: f64) {
    let mut e = find_effect("transform").expect("transform").instance();
    e.params.get_mut("use_comp_shutter").expect("comp").value = ParamValue::Bool(comp);
    e.params.get_mut("shutter_angle").expect("angle").value = ParamValue::Float(angle);
    let sc = e.params.get_mut("scale_height").expect("scale");
    sc.put_keyframe(R.tick_of(k0), ParamValue::Float(from));
    sc.put_keyframe(R.tick_of(k1), ParamValue::Float(to));
    let it = clip(s, c);
    let pos = it.effects.iter().position(|e| e.def().is_some_and(|d| d.intrinsic)).unwrap_or(it.effects.len());
    it.effects.insert(pos, e);
}

fn render(s: &Scene, frame: i64) -> Image {
    render_sequence(&s.p, s.seq, R.tick_of(frame), RenderOptions::default(), &s.map)
}

/// Pixels of the middle row that are neither black nor white: the width of the box's edges
/// (a sharp edge has at most one partial pixel per side).
fn soft_pixels(img: &Image) -> usize {
    (0..img.w).filter(|x| (0.03..0.97).contains(&img.get(*x, img.h / 2)[0])).count()
}

/// Columns the box's left edge covers (first column above half white) on the middle row.
fn left_edge(img: &Image) -> Option<usize> {
    (0..img.w).find(|x| img.get(*x, img.h / 2)[0] > 0.5)
}

#[test]
fn moving_push_in_smears_edges_by_the_shutter_travel() {
    // 100 → 300 % over frames 9..15: at frame 12 the box edges (48 px from the centre at 100 %)
    // travel 48 × 200 % / 6 = 16 px a frame, 8 px while a 180° shutter is open
    let (mut sharp, c) = scene(true);
    push(&mut sharp, c, 100.0, 300.0, 9, 15, false, 0.0);
    let (mut blurred, c) = scene(true);
    push(&mut blurred, c, 100.0, 300.0, 9, 15, false, 180.0);
    let (a, b) = (render(&sharp, 12), render(&blurred, 12));
    let (sa, sb) = (soft_pixels(&a), soft_pixels(&b));
    eprintln!("soft pixels: none {sa}, 180° {sb}");
    // (the 200 % upscale itself spreads each edge over ~2 px)
    assert!(sa <= 4, "no shutter: sharp edges, got {sa} soft pixels");
    // two edges, each ~8 px of ramp
    assert!((sa + 10..=sa + 18).contains(&sb), "180° shutter: ~16 more soft pixels, got {sb} (vs {sa})");
    // the blur is centred on the frame's own position
    let (ea, eb) = (left_edge(&a).expect("edge"), left_edge(&b).expect("edge"));
    assert!(ea.abs_diff(eb) <= 1, "blur centred on the frame time: {ea} vs {eb}");
    // 360° doubles the smear
    let (mut wide, c) = scene(true);
    push(&mut wide, c, 100.0, 300.0, 9, 15, false, 360.0);
    let sw = soft_pixels(&render(&wide, 12));
    eprintln!("soft pixels: 360° {sw}");
    assert!(sw >= sb + 10, "360° smears about twice as far as 180°: {sw} vs {sb}");
    // deterministic
    assert_eq!(render(&blurred, 12).px, b.px);
}

#[test]
fn transform_on_a_clip_blurs_too() {
    // the full-frame matte's own Transform shrinks it to a box, 30 → 90 % over frames 9..15:
    // its edges travel 160 × 60 % / 6 = 16 px a frame
    let (mut s, c) = scene(false);
    push(&mut s, c, 30.0, 90.0, 9, 15, false, 0.0);
    let sharp = soft_pixels(&render(&s, 12));
    let (mut s, c) = scene(false);
    push(&mut s, c, 30.0, 90.0, 9, 15, false, 180.0);
    let soft = soft_pixels(&render(&s, 12));
    eprintln!("clip: {sharp} → {soft}");
    assert!(sharp <= 2, "{sharp}");
    assert!(soft >= 10, "a clip's own Transform motion-blurs: {soft} soft pixels");
}

#[test]
fn static_or_closed_shutter_renders_exactly_without_blur() {
    let (mut reference, c) = scene(true);
    push(&mut reference, c, 100.0, 300.0, 9, 15, false, 0.0);
    // holds before and after the move: nothing moves while the shutter is open
    let (mut held, c) = scene(true);
    push(&mut held, c, 100.0, 300.0, 9, 15, false, 180.0);
    for f in [3, 20, 40] {
        assert_eq!(render(&held, f).px, render(&reference, f).px, "frame {f}: a static Transform is not blurred");
    }
    // the composition's shutter angle (0°, as in Premiere) wins over the effect's own angle
    let (mut comp, c) = scene(true);
    push(&mut comp, c, 100.0, 300.0, 9, 15, true, 180.0);
    assert_eq!(render(&comp, 12).px, render(&reference, 12).px, "Use Composition's Shutter Angle: no blur");
    // a frame hold with held filters freezes the Transform: no blur
    let (mut frozen, c) = scene(false);
    push(&mut frozen, c, 30.0, 90.0, 9, 15, false, 180.0);
    let it = clip(&mut frozen, c);
    it.frame_hold = Some(R.tick_of(12));
    it.hold_filters = true;
    assert!(soft_pixels(&render(&frozen, 30)) <= 2);
}

#[test]
fn hostile_shutter_values_do_not_crash() {
    for angle in [f64::NAN, f64::INFINITY, -50.0, 1e12] {
        let (mut s, c) = scene(true);
        push(&mut s, c, 100.0, 1e9, 9, 15, false, angle);
        let img = render(&s, 12);
        assert!(img.px.iter().all(|v| v.is_finite()), "angle {angle}");
    }
}

#[test]
fn eased_push_in_blurs_most_mid_move() {
    // a push-in with Bezier (ease in and out) keys and a 180° shutter (exaggerated, 100 → 300 %
    // over half a second, so the smear is many pixels wide): the move is fastest in the middle,
    // so the smear peaks there and vanishes at the stationary ends
    let (mut s, c) = scene(true);
    push(&mut s, c, 100.0, 300.0, 0, 12, false, 180.0);
    if let Some(e) = clip(&mut s, c).effects.iter_mut().find(|e| e.effect == "transform")
        && let Some(p) = e.params.get_mut("scale_height")
    {
        for k in &mut p.keyframes {
            k.interp = filmcraft_project::Interpolation::Bezier;
        }
    }
    let soft: Vec<usize> = [0, 3, 6, 9, 12].iter().map(|f| soft_pixels(&render(&s, *f))).collect();
    eprintln!("eased: {soft:?}");
    assert!(soft[2] >= soft[0] + 8 && soft[2] >= soft[4] + 8, "the smear peaks mid-move: {soft:?}");
    assert!(soft[1] > soft[0] && soft[3] > soft[4], "and grows / shrinks with the speed: {soft:?}");
}
