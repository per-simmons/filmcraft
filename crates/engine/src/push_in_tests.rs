//! Pat's push-ins, end to end through engine commands: an adjustment layer over a range,
//! Transform Scale keyframes eased in and out (Premiere's Ease In + Ease Out = flat Bezier
//! handles), motion blur from the Transform's own shutter angle, and the strong push-in that
//! reframes the subject onto a third with a title in the open side. Both the step-by-step command
//! sequence and the one-shot `clip.pushIn` build the same editable model.

use filmcraft_project::{ClipId, Interpolation, TrackItem};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::Session;

/// A 640×360, 60 fps sequence with a white box (a matte at 30 % Motion scale: 192 × 108 px,
/// centred) on black on V1, four seconds long.
fn session() -> Session {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"name": "Push", "width": 640, "height": 360, "fps": 60.0})).unwrap();
    let r = s.execute("file.newColorMatte", json!({"color": "#ffffff", "seconds": 4.0, "width": 640, "height": 360})).unwrap();
    let m = r["item"].as_u64().unwrap();
    let r = s.execute("timeline.place", json!({"item": m, "track": "V1", "seconds": 0.0})).unwrap();
    let c = r["clips"][0].as_u64().unwrap();
    s.execute("effects.setParam", json!({"clip": c, "effect": "motion", "param": "scale", "value": 30.0})).unwrap();
    s
}

fn clip(s: &Session, c: u64) -> TrackItem {
    s.active_sequence().unwrap().find_item(ClipId(c)).unwrap().1.clone()
}

/// The Transform's Scale at sequence frame `f` (60 fps).
fn scale_at(s: &Session, c: u64, f: i64) -> f64 {
    let it = clip(s, c);
    let t = s.sequence_rate().tick_of(f);
    it.effect("transform").unwrap().param("scale_height").unwrap().f64_at(it.source_time_at(t))
}

fn render(s: &Session, f: i64) -> filmcraft_render::Image {
    s.render_program_at(1.0, s.sequence_rate().tick_of(f)).unwrap()
}

/// Neither-black-nor-white pixels on the middle row: the soft width of the box's edges.
fn soft(img: &filmcraft_render::Image) -> usize {
    (0..img.w).filter(|x| (0.03..0.97).contains(&img.get(*x, img.h / 2)[0])).count()
}

/// Premiere's own Ease In + Ease Out samples of Pat's first push (100 → 108 % over frames 24..69
/// at 60 fps), read from the live sequence (push-ins.md, "Professional-finish confirmation").
const PREMIERE: [(i64, f64); 7] = [(25, 100.0117), (30, 100.3887), (46, 103.8667), (64, 107.7257), (68, 107.9883), (69, 108.0), (120, 108.0)];

fn assert_premiere_curve(s: &Session, c: u64) {
    for (f, want) in PREMIERE {
        let got = scale_at(s, c, f);
        assert!((got - want).abs() < 0.02, "frame {f}: {got:.4} (Premiere {want})");
    }
    assert_eq!(scale_at(s, c, 0), 100.0, "holds before the move");
    assert_eq!(scale_at(s, c, 24), 100.0);
}

#[test]
fn pats_push_in_step_by_step() {
    let mut s = session();
    // New Adjustment Layer, dragged onto V2 at the start (Pat's Cmd+Y, drag, extend)
    let r = s.execute("file.newAdjustmentLayer", json!({"seconds": 4.0})).unwrap();
    let item = r["item"].as_u64().unwrap();
    let r = s.execute("timeline.place", json!({"item": item, "track": "V2", "seconds": 0.0})).unwrap();
    let adj = r["clips"][0].as_u64().unwrap();
    // Transform, composition shutter off, 120° shutter
    s.execute("effects.apply", json!({"clips": [adj], "effect": "transform"})).unwrap();
    let set = |s: &mut Session, param: &str, value: Value| {
        s.execute("effects.setParam", json!({"clip": adj, "effect": "transform", "param": param, "value": value})).unwrap();
    };
    set(&mut s, "use_comp_shutter", json!(false));
    set(&mut s, "shutter_angle", json!(120.0));
    // Scale keys: 100 at frame 24 (stopwatch on there), 108 at frame 69
    s.execute("effects.setParam", json!({"clip": adj, "effect": "transform", "param": "scale_height", "value": 100.0, "frame": 24, "keyframe": true})).unwrap();
    s.execute("effects.setParam", json!({"clip": adj, "effect": "transform", "param": "scale_height", "value": 108.0, "frame": 69})).unwrap();
    // select every Scale key, Ease In, then Ease Out (as in Premiere)
    for ease in ["easeIn", "easeOut"] {
        s.execute("effects.setInterpolation", json!({"clip": adj, "effect": "transform", "param": "scale_height", "all": true, "interpolation": ease}))
            .unwrap();
    }
    let it = clip(&s, adj);
    let keys = &it.effect("transform").unwrap().param("scale_height").unwrap().keyframes;
    assert_eq!(keys.len(), 2);
    assert!(keys.iter().all(|k| k.interp == Interpolation::Bezier), "Ease In + Ease Out = eased on both sides");
    assert_premiere_curve(&s, adj);

    // motion blur while the push moves (a subtle 8 % push: under a pixel of smear, so compare
    // pictures), none on the holds
    let (moving, held, before) = (render(&s, 46), render(&s, 100), render(&s, 10));
    set(&mut s, "shutter_angle", json!(0.0));
    assert_ne!(moving.px, render(&s, 46).px, "120° shutter blurs the move");
    assert_eq!(held.px, render(&s, 100).px, "the hold is not blurred");
    assert_eq!(before.px, render(&s, 10).px, "nor the frames before the move");
}

#[test]
fn push_in_command_builds_the_same_editable_push() {
    let mut s = session();
    let r = s
        .execute(
            "clip.pushIn",
            json!({"startSeconds": 0.0, "endSeconds": 4.0, "moveStartFrame": 24, "moveEndFrame": 69, "from": 100.0, "to": 108.0, "shutterAngle": 120.0}),
        )
        .unwrap();
    assert_eq!(r["created"], true);
    let adj = r["clip"].as_u64().unwrap();
    let q = s.active_sequence().unwrap();
    assert!(q.video_tracks[1].items.iter().any(|i| i.id.0 == adj), "the adjustment layer lands on V2, the first free track above V1");
    let it = clip(&s, adj);
    assert_eq!((it.start, it.end()), (Tick::ZERO, s.sequence_rate().tick_of(240)));
    let e = it.effect("transform").unwrap();
    assert_eq!(e.param("use_comp_shutter").unwrap().value.as_bool(), Some(false));
    assert_eq!(e.param("shutter_angle").unwrap().value.as_f64(), Some(120.0));
    assert_premiere_curve(&s, adj);
    // it is an ordinary keyframed Transform: change the destination like any other key
    let it = clip(&s, adj);
    let end = it.source_time_at(s.sequence_rate().tick_of(69));
    s.execute("effects.setKeyframe", json!({"clip": adj, "effect": "transform", "param": "scale_height", "mediaTime": end.0, "value": 110.0})).unwrap();
    assert!((scale_at(&s, adj, 100) - 110.0).abs() < 1e-9);
    // a second move on the same layer (back out) reuses its Transform and starts where it is
    let r = s.execute("clip.pushIn", json!({"clip": adj, "moveStartFrame": 150, "moveEndFrame": 190, "to": 100.0})).unwrap();
    assert_eq!(r["created"], false);
    assert_eq!(clip(&s, adj).effects.iter().filter(|e| e.effect == "transform").count(), 1);
    assert!((scale_at(&s, adj, 150) - 110.0).abs() < 1e-9 && (scale_at(&s, adj, 200) - 100.0).abs() < 1e-9);
    // motion blur on the way: an exaggerated push (100 → 200 % in 6 frames) smears the box edges
    let r = s.execute("clip.pushIn", json!({"clip": adj, "moveStartFrame": 200, "moveEndFrame": 206, "from": 100.0, "to": 200.0})).unwrap();
    assert_eq!(r["created"], false);
    let (moving, held) = (soft(&render(&s, 203)), soft(&render(&s, 220)));
    eprintln!("soft pixels: moving {moving}, held {held}");
    assert!(moving >= held + 6, "{moving} vs {held}");
}

#[test]
fn push_in_with_text_beside_the_subject() {
    let mut s = session();
    // reframe the box (the "face") onto the left third while pushing to 150 %: with the anchor at
    // the frame centre, rendered = Position + scale × (input − Anchor), so Position = the target
    let third = 640.0 / 3.0;
    let r = s
        .execute(
            "clip.pushIn",
            json!({"startSeconds": 0.0, "endSeconds": 4.0, "moveStartSeconds": 0.5, "moveEndSeconds": 1.25, "to": 150.0, "position": [third, 180.0]}),
        )
        .unwrap();
    let adj = r["clip"].as_u64().unwrap();
    // the title in the open right side as the push lands, on a track above the adjustment layer
    let land = s.sequence_rate().tick_of(75);
    let r = s.execute("graphics.newText", json!({"text": "HELLO", "position": [440.0, 200.0], "size": 60.0, "seconds": 2.0, "time": land.0})).unwrap();
    let title = r["clip"].as_u64().or_else(|| r.as_u64()).unwrap_or_else(|| panic!("{r}"));
    let q = s.active_sequence().unwrap();
    let track_of = |c: u64| q.video_tracks.iter().position(|t| t.items.iter().any(|i| i.id.0 == c)).unwrap();
    assert!(track_of(title) > track_of(adj), "the title sits above the adjustment layer, so the push does not zoom it");

    let pos = clip(&s, adj).effect("transform").unwrap().param("position").unwrap().clone();
    assert_eq!(pos.keyframes.len(), 2);
    assert!(pos.keyframes.iter().all(|k| k.interp == Interpolation::Bezier));
    // at the hold the box is 288 × 162 px, centred on the left third (row 110: inside the box,
    // above the title's letters)
    let img = render(&s, 120);
    let row: Vec<usize> = (0..img.w).filter(|x| img.get(*x, 110)[0] > 0.5).collect();
    let (l, r) = (*row.first().unwrap(), *row.last().unwrap());
    assert!(((l + r) as f64 / 2.0 - third).abs() < 2.0 && (r + 1 - l).abs_diff(288) <= 2, "box at {l}..{r}");
    // the title is untouched by the push: identical to the frame without the adjustment layer in
    // its open-side region
    s.execute("clip.enable", json!({"clips": [adj]})).unwrap();
    let plain = render(&s, 120);
    let region = |i: &filmcraft_render::Image| (150..260).flat_map(|y| (400..640).map(move |x| (x, y))).map(|(x, y)| i.get(x, y)).collect::<Vec<_>>();
    let (a, b) = (region(&img), region(&plain));
    assert!(a.iter().any(|p| p[0] > 0.5), "the title is there");
    assert_eq!(a, b, "the push-in does not move or scale the title");
}

#[test]
fn push_in_refuses_hostile_parameters() {
    let mut s = session();
    for bad in [
        json!({}),
        json!({"startSeconds": 1.0, "endSeconds": 0.5}),
        json!({"startSeconds": 0.0, "endSeconds": 4.0, "to": f64::MAX}),
        json!({"startSeconds": 0.0, "endSeconds": 4.0, "to": -5.0}),
        json!({"startSeconds": 0.0, "endSeconds": 4.0, "shutterAngle": 720.0}),
        json!({"startSeconds": 0.0, "endSeconds": 4.0, "ease": "bounce"}),
        json!({"startSeconds": 0.0, "endSeconds": 4.0, "position": [1.0]}),
        json!({"startSeconds": 0.0, "endSeconds": 4.0, "moveSeconds": -1.0}),
        json!({"startSeconds": 0.0, "endSeconds": 4.0, "moveStartSeconds": 3.9, "moveEndSeconds": 9.0}),
        json!({"startSeconds": 0.0, "endSeconds": 4.0, "track": "V1"}),
        json!({"startSeconds": 0.0, "endSeconds": 4.0, "track": "V9"}),
        json!({"clip": 999_999}),
    ] {
        assert!(s.execute("clip.pushIn", bad.clone()).is_err(), "{bad}");
    }
    // nothing half-made by a refused call before the layer exists
    assert!(s.active_sequence().unwrap().video_tracks.iter().skip(1).all(|t| t.items.is_empty()));
}
