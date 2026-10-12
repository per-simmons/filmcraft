//! Headless UI tests of scrolling the Timeline: what the wheel does with each modifier, how far
//! the view can go, and the zoom scroll bar. Premiere Pro 26.5.2 on macOS, checked in the app: the
//! wheel moves the tracks up and down, Cmd + wheel moves the Timeline sideways, Option + wheel
//! zooms (FilmCraft: over the ruler; over the tracks it resizes the track under the pointer unless
//! Settings ▸ Timeline ▸ "Option + wheel over the tracks" is "zoom"), Option + wheel over a track
//! header resizes that track, and the view stops ten minutes past the end of the sequence.

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use filmcraft_ui_egui::panels::timeline::{max_scroll, timeline_extent};
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
}

impl Driver {
    fn demo() -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let harness = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000).build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx };
        d.frames(4);
        d
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            let ctx = self.harness.ctx.clone();
            let mut raw = std::mem::take(self.harness.input_mut());
            eframe::App::raw_input_hook(self.harness.state_mut(), &ctx, &mut raw);
            *self.harness.input_mut() = raw;
            self.harness.step();
        }
    }

    fn call(&mut self, method: &str, params: Value) -> Value {
        let (req, reply) = ControlRequest::new(method, params.clone());
        self.tx.send(req).unwrap();
        for _ in 0..600 {
            self.frames(1);
            if let Ok(v) = reply.try_recv() {
                return v;
            }
        }
        panic!("no reply to {method} {params}");
    }

    fn ok(&mut self, method: &str, params: Value) -> Value {
        let v = self.call(method, params.clone());
        assert_eq!(v["ok"], json!(true), "{method} {params} failed: {v}");
        v["result"].clone()
    }

    fn exec(&mut self, command: &str, params: Value) -> Value {
        self.ok("engine.execute", json!({"command": command, "params": params}))
    }
}

impl Driver {
    fn app(&mut self) -> &mut FilmcraftApp {
        self.harness.state_mut()
    }

    fn rect(&mut self, id: &str) -> [f32; 4] {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        let e = v.as_array().unwrap().iter().find(|e| e["id"] == json!(id)).unwrap_or_else(|| panic!("no element {id}")).clone();
        let r = e["rect"].as_array().unwrap();
        [0, 1, 2, 3].map(|k| r[k].as_f64().unwrap() as f32)
    }

    /// Turn the wheel over the tracks (`fy`: 0.2 is among the video tracks, 0.8 among the audio
    /// tracks), with modifiers.
    fn wheel(&mut self, fy: f64, dx: f64, dy: f64, modifiers: Value) {
        self.ok("ui.scroll", json!({"id": "timeline.tracks", "fx": 0.5, "fy": fy, "dx": dx, "dy": dy, "modifiers": modifiers}));
        self.frames(3);
    }

    /// (time at the left edge, zoom, video scroll, audio scroll), once the view has settled.
    fn view(&mut self) -> (f64, f64, f32, f32) {
        for _ in 0..200 {
            let v = &self.app().ui.timeline;
            if v.pps == v.target_pps && v.scroll == v.target_scroll {
                break;
            }
            self.frames(1);
        }
        let v = &self.app().ui.timeline;
        (v.target_scroll, v.target_pps, v.v_scroll, v.a_scroll)
    }

    /// The demo sequence with enough tracks to scroll both halves, at a known zoom.
    fn tall() -> Driver {
        let mut d = Driver::demo();
        d.exec("sequence.addTracks", json!({"video": 12, "audio": 12}));
        d.ok("ui.set", json!({"timeline": {"pps": 50.0, "scroll": 4.0}}));
        d.frames(4);
        assert_eq!(d.view(), (4.0, 50.0, 0.0, 0.0));
        d
    }

    fn seconds(&mut self) -> f64 {
        self.app().session.active_sequence().unwrap().duration().seconds()
    }
}

fn none() -> Value {
    json!({})
}

#[test]
fn the_wheel_moves_the_tracks_up_and_down() {
    let mut d = Driver::tall();
    // over the video tracks: wheeling up brings the higher tracks in (Video 1 is at the bottom)
    d.wheel(0.2, 0.0, 40.0, none());
    assert_eq!(d.view(), (4.0, 50.0, 40.0, 0.0), "the video tracks moved, nothing else");
    d.wheel(0.2, 0.0, -25.0, none());
    assert_eq!(d.view(), (4.0, 50.0, 15.0, 0.0));
    // over the audio tracks: wheeling down brings the lower tracks in
    d.wheel(0.8, 0.0, -30.0, none());
    assert_eq!(d.view(), (4.0, 50.0, 15.0, 30.0));
    // neither half goes past its first or last track
    d.wheel(0.2, 0.0, -1e6, none());
    d.wheel(0.8, 0.0, 1e6, none());
    assert_eq!(d.view(), (4.0, 50.0, 0.0, 0.0));
    d.wheel(0.8, 0.0, -1e6, none());
    let (_, _, _, lowest) = d.view();
    assert!(lowest > 100.0 && lowest < 15.0 * 200.0, "{lowest}");
}

#[test]
fn cmd_wheel_moves_the_timeline_sideways_and_option_wheel_zooms() {
    let mut d = Driver::tall();
    // Cmd + wheel: sideways, a point of wheel for a point of Timeline; the tracks stay
    d.wheel(0.2, 0.0, -100.0, json!({"command": true}));
    assert_eq!(d.view(), (6.0, 50.0, 0.0, 0.0), "100 points at 50 points a second: 2 s later");
    d.wheel(0.8, 0.0, 250.0, json!({"command": true}));
    assert_eq!(d.view(), (1.0, 50.0, 0.0, 0.0));
    d.wheel(0.8, 0.0, 1e5, json!({"command": true}));
    assert_eq!(d.view().0, 0.0, "not before the start");
    // Option + wheel: zoom about the pointer; it is not what Cmd + wheel does (over the tracks
    // with the setting at Premiere's "zoom")
    d.app().session.prefs.timeline.option_wheel_over_tracks = "zoom".into();
    d.ok("ui.set", json!({"timeline": {"scroll": 4.0}}));
    d.frames(3);
    let tracks = d.rect("timeline.tracks");
    let under_pointer = 4.0 + (tracks[2] as f64 * 0.5) / 50.0;
    d.wheel(0.2, 0.0, 20.0, json!({"alt": true}));
    let (scroll, pps, v, a) = d.view();
    assert!((pps - 60.0).abs() < 1e-6, "zoomed in by a fifth: {pps}");
    assert!((scroll + (tracks[2] as f64 * 0.5) / pps - under_pointer).abs() < 0.02, "the time under the pointer stays there: {scroll}");
    assert_eq!((v, a), (0.0, 0.0));
    d.wheel(0.2, 0.0, -20.0, json!({"alt": true}));
    assert!((d.view().1 - 48.0).abs() < 1e-6, "and out again");
    // a sideways gesture (trackpad swipe, tilt wheel) is sideways whatever is held; so is Shift
    d.ok("ui.set", json!({"timeline": {"pps": 50.0, "scroll": 4.0}}));
    d.frames(3);
    d.wheel(0.2, -50.0, 5.0, none());
    assert_eq!(d.view(), (5.0, 50.0, 0.0, 0.0));
    d.wheel(0.2, 0.0, -50.0, json!({"shift": true}));
    assert_eq!(d.view(), (6.0, 50.0, 0.0, 0.0));
}

/// Settings ▸ Timeline ▸ Timeline Mouse Scrolling "Horizontal": the wheel moves the Timeline
/// sideways and Cmd + wheel moves the tracks.
#[test]
fn the_mouse_scrolling_setting_swaps_wheel_and_cmd_wheel() {
    let mut d = Driver::tall();
    assert_eq!(d.app().session.prefs.timeline.mouse_scrolling, "vertical", "Premiere's default on macOS");
    d.app().session.prefs.timeline.mouse_scrolling = "horizontal".into();
    d.wheel(0.2, 0.0, -100.0, none());
    assert_eq!(d.view(), (6.0, 50.0, 0.0, 0.0));
    d.wheel(0.2, 0.0, 40.0, json!({"command": true}));
    assert_eq!(d.view(), (6.0, 50.0, 40.0, 0.0));
    d.ok("ui.scroll", json!({"id": "timeline.ruler", "fx": 0.5, "fy": 0.5, "dy": 20.0, "modifiers": {"alt": true}}));
    d.frames(3);
    assert!((d.view().1 - 60.0).abs() < 1e-6, "Option + wheel over the ruler zooms either way");
}

/// The view cannot leave the Timeline: scrolled 39 minutes past a 26 second sequence, nothing was
/// in view and the scroll bar's thumb sat at the right end whatever was done to it.
#[test]
fn the_view_stops_ten_minutes_past_the_end_of_the_sequence() {
    let mut d = Driver::tall();
    let seconds = d.seconds();
    let width = d.rect("timeline.tracks")[2];
    assert_eq!(timeline_extent(seconds), seconds + 600.0);
    let limit = max_scroll(seconds, width, 50.0);
    assert!((limit - (seconds + 600.0 - width as f64 / 50.0)).abs() < 1e-9);
    // by the wheel
    d.wheel(0.2, 0.0, -1e7, json!({"command": true}));
    assert!((d.view().0 - limit).abs() < 1e-6, "{} vs {limit}", d.view().0);
    d.wheel(0.2, 0.0, -500.0, json!({"command": true}));
    assert!((d.view().0 - limit).abs() < 1e-6);
    // by anything else that moves the view (a script here: the state that was reported stuck)
    d.ok("ui.set", json!({"timeline": {"pps": 2.9863201066269047, "scroll": 2325.538505030118}}));
    d.frames(4);
    let (scroll, pps, ..) = d.view();
    assert!((scroll - max_scroll(seconds, width, pps)).abs() < 1e-6 && scroll < 400.0, "{scroll}");
    // zoomed out until the whole Timeline is in view, there is nowhere to scroll to
    d.ok("ui.set", json!({"timeline": {"pps": 0.5, "scroll": 50.0}}));
    d.frames(4);
    assert_eq!(d.view().0, 0.0);
    // numbers that are not numbers do no harm
    d.app().ui.timeline.target_scroll = f64::NAN;
    d.app().ui.timeline.scroll = f64::INFINITY;
    d.frames(3);
    assert_eq!(d.view().0, 0.0);
}

#[test]
fn the_scroll_bar_thumb_follows_the_pointer_and_reaches_both_ends() {
    let mut d = Driver::tall();
    let seconds = d.seconds();
    let track = d.rect("timeline.zoomBar.track");
    let limit = max_scroll(seconds, d.rect("timeline.tracks")[2], 50.0);
    // at the start the thumb is at the left end of the bar, and wide enough to grab
    d.ok("ui.set", json!({"timeline": {"scroll": 0.0}}));
    d.frames(3);
    let thumb = d.rect("timeline.zoomBar");
    assert!((thumb[0] - track[0]).abs() < 0.5 && thumb[2] >= 30.0 && thumb[2] < track[2] / 4.0, "{thumb:?} in {track:?}");
    // dragged by its middle it moves with the pointer, and the view with it
    let y = thumb[1] + thumb[3] / 2.0;
    let grab = thumb[0] + thumb[2] / 2.0;
    d.ok("ui.drag", json!({"from": {"x": grab, "y": y}, "to": {"x": grab + 300.0, "y": y}, "steps": 10}));
    d.frames(3);
    let moved = d.rect("timeline.zoomBar");
    assert!((moved[0] - thumb[0] - 300.0).abs() < 2.0, "the thumb went {} points for a 300 point drag", moved[0] - thumb[0]);
    let travel = (track[2] - thumb[2]) as f64;
    assert!((d.view().0 - 300.0 / travel * limit).abs() < limit * 0.01, "{}", d.view().0);
    assert_eq!(d.view().1, 50.0, "a drag by the middle does not zoom");
    // dragged past the right end it stops there: the view is at its limit
    let grab = moved[0] + moved[2] / 2.0;
    d.ok("ui.drag", json!({"from": {"x": grab, "y": y}, "to": {"x": track[0] + track[2] + 400.0, "y": y}, "steps": 10}));
    d.frames(3);
    let right = d.rect("timeline.zoomBar");
    assert!((right[0] + right[2] - (track[0] + track[2])).abs() < 0.5, "{right:?}");
    assert!((d.view().0 - limit).abs() < 1e-6);
    // and from the right end it comes back: dragged left it leaves the end and follows again
    let grab = right[0] + right[2] / 2.0;
    d.ok("ui.drag", json!({"from": {"x": grab, "y": y}, "to": {"x": grab - 250.0, "y": y}, "steps": 10}));
    d.frames(3);
    let back = d.rect("timeline.zoomBar");
    assert!((right[0] - back[0] - 250.0).abs() < 2.0, "{back:?} from {right:?}");
    // all the way left: the start of the sequence
    let grab = back[0] + back[2] / 2.0;
    d.ok("ui.drag", json!({"from": {"x": grab, "y": y}, "to": {"x": track[0] - 500.0, "y": y}, "steps": 10}));
    d.frames(3);
    assert_eq!(d.view().0, 0.0);
    assert!((d.rect("timeline.zoomBar")[0] - track[0]).abs() < 0.5);
    // a click on the bar beside the thumb brings the thumb there
    let target = track[0] + track[2] * 0.6;
    d.ok("ui.click", json!({"x": target, "y": y}));
    d.frames(3);
    let jumped = d.rect("timeline.zoomBar");
    assert!((jumped[0] + jumped[2] / 2.0 - target).abs() < 2.0, "{jumped:?}");
    // the handles at its ends still zoom: the right one out, the left edge of the view staying
    let at = d.view().0;
    d.ok("ui.drag", json!({"from": {"x": jumped[0] + jumped[2] - 3.0, "y": y}, "to": {"x": jumped[0] + jumped[2] + 60.0, "y": y}, "steps": 8}));
    d.frames(3);
    let (scroll, pps, ..) = d.view();
    assert!(pps < 50.0 && (scroll - at).abs() < 1e-6, "{pps} {scroll}");
}

impl Driver {
    /// Turn the wheel over element `id` (at its middle) with modifiers.
    fn wheel_on(&mut self, id: &str, dy: f64, modifiers: Value) {
        self.ok("ui.scroll", json!({"id": id, "fx": 0.5, "fy": 0.5, "dx": 0.0, "dy": dy, "modifiers": modifiers}));
        self.frames(3);
    }

    /// Heights of the tracks named (V1, A2…) the Timeline lays out.
    fn heights(&mut self, labels: &[&str]) -> Vec<f32> {
        let seq = self.app().session.active_sequence().unwrap().clone();
        labels
            .iter()
            .map(|l| {
                let i: usize = l[1..].parse().unwrap();
                let video = l.starts_with('V');
                let id = if video { seq.video_tracks[i - 1].id.0 } else { seq.audio_tracks[i - 1].id.0 };
                self.app().ui.timeline.track_height(id, video)
            })
            .collect()
    }
}

/// Option + wheel over a track (its clips or its header) makes that track taller or shorter, a
/// step proportional to its height per notch, and nothing else moves: no zoom, no scroll.
#[test]
fn option_wheel_over_a_track_resizes_that_track_only() {
    let mut d = Driver::demo();
    d.ok("ui.set", json!({"timeline": {"pps": 50.0, "scroll": 4.0}}));
    d.frames(4);
    let before = d.view();
    let h0 = d.heights(&["V1", "V2", "A1", "A2"]);
    // over A1's lane, wheel up: A1 grows by about a fifth, the rest stay
    d.wheel_on("timeline.track.A1.row", 40.0, json!({"alt": true}));
    let h1 = d.heights(&["V1", "V2", "A1", "A2"]);
    assert!(h1[2] > h0[2] * 1.18 && h1[2] < h0[2] * 1.26, "A1 {} → {}", h0[2], h1[2]);
    assert_eq!((h1[0], h1[1], h1[3]), (h0[0], h0[1], h0[3]), "only A1");
    assert_eq!(d.view(), before, "no zoom and no scroll while Option is held over a track");
    // over V1's header, wheel down: V1 shrinks
    d.wheel_on("timeline.track.V1.header", -40.0, json!({"alt": true}));
    let h2 = d.heights(&["V1", "V2", "A1", "A2"]);
    assert!(h2[0] < h0[0] * 0.85, "V1 {} → {}", h0[0], h2[0]);
    assert_eq!((h2[1], h2[2], h2[3]), (h1[1], h1[2], h1[3]));
    assert_eq!(d.view(), before);
    // Cmd as well: finer steps
    d.wheel_on("timeline.track.A2.row", 40.0, json!({"alt": true, "command": true}));
    let h3 = d.heights(&["A2"])[0];
    assert!(h3 > h0[3] && h3 < h0[3] * 1.08, "fine: A2 {} → {h3}", h0[3]);
    assert_eq!(d.view(), before);
    // Shift + Option: every audio track takes A1's new height; the video tracks stay
    d.wheel_on("timeline.track.A1.row", 40.0, json!({"alt": true, "shift": true}));
    let h4 = d.heights(&["V1", "V2", "A1", "A2"]);
    assert!(h4[2] > h1[2] && (h4[2] - h4[3]).abs() < 0.01, "all audio tracks: {h4:?}");
    assert_eq!((h4[0], h4[1]), (h2[0], h2[1]));
    // Shift + wheel over a header (Premiere): every track of its kind
    d.wheel_on("timeline.track.V2.header", 40.0, json!({"shift": true}));
    let h5 = d.heights(&["V1", "V2"]);
    assert!(h5[1] > h2[1] && (h5[0] - h5[1]).abs() < 0.01, "all video tracks: {h5:?}");
    assert_eq!(d.view(), before);
    // clamped: a big spin never goes past the largest or under the smallest height
    let seq = d.app().session.state.active_sequence.unwrap();
    let a1 = d.app().session.active_sequence().unwrap().audio_tracks[0].id.0;
    d.wheel_on("timeline.track.A1.row", 1e6, json!({"alt": true}));
    assert_eq!(d.app().ui.timeline.track_height(a1, false), filmcraft_ui_egui::state::TRACK_H_MAX);
    d.wheel_on("timeline.track.A1.row", -1e6, json!({"alt": true}));
    assert_eq!(d.app().ui.timeline.track_height(a1, false), filmcraft_ui_egui::state::TRACK_H_MIN);
    // the heights are kept with the sequence's view (saved beside the project)
    d.frames(2);
    let kept = d.app().session.state.timeline_views.get(&seq).cloned().expect("the view is kept");
    assert_eq!(kept.track_heights.get(&a1).copied(), Some(filmcraft_ui_egui::state::TRACK_H_MIN));
    // Option + wheel over the ruler still zooms
    d.wheel_on("timeline.ruler", 20.0, json!({"alt": true}));
    assert!((d.view().1 - 60.0).abs() < 1e-6, "zoom over the ruler: {:?}", d.view());
    // the setting at "zoom": over the clips Option + wheel zooms, over a header it still resizes
    d.app().session.prefs.timeline.option_wheel_over_tracks = "zoom".into();
    let a2 = d.heights(&["A2"])[0];
    d.wheel_on("timeline.track.A2.row", 20.0, json!({"alt": true}));
    assert!((d.view().1 - 72.0).abs() < 1e-6 && d.heights(&["A2"])[0] == a2);
    d.wheel_on("timeline.track.A2.header", 40.0, json!({"alt": true}));
    assert!(d.heights(&["A2"])[0] > a2);
}

/// `timeline.setTrackHeight`: what Option + wheel does, for agents.
#[test]
fn set_track_height_is_a_command() {
    let mut d = Driver::demo();
    let r = d.exec("timeline.setTrackHeight", json!({"track": "A1", "height": 150.0}));
    assert_eq!((r["track"].as_str(), r["height"].as_f64()), (Some("A1"), Some(150.0)));
    d.frames(3);
    assert_eq!(d.heights(&["A1"])[0], 150.0);
    // and the Timeline lays A1 out that tall (the audio half scrolled to the top)
    d.app().ui.timeline.split = 0.2;
    d.frames(3);
    assert!((d.rect("timeline.track.A1.row")[3] - 150.0).abs() < 0.01);
    let r = d.exec("timeline.setTrackHeight", json!({"track": "A1", "wheel": -40.0}));
    assert!(r["height"].as_f64().unwrap() < 125.0);
    let r = d.exec("timeline.setTrackHeight", json!({"track": "V1", "height": 90.0, "all": true}));
    assert_eq!(r["videoTrackHeight"].as_f64(), Some(90.0));
    // hostile parameters: an error, never a crash
    for bad in [
        json!({}),
        json!({"track": "Z9", "height": 50.0}),
        json!({"track": "A99", "height": 50.0}),
        json!({"track": "A1"}),
        json!({"track": [1], "height": 1.0}),
    ] {
        let v = d.call("engine.execute", json!({"command": "timeline.setTrackHeight", "params": bad}));
        assert_eq!(v["ok"], json!(false), "{bad} should fail: {v}");
    }
    let r = d.exec("timeline.setTrackHeight", json!({"track": "A1", "height": 1e30}));
    assert_eq!(r["height"].as_f64(), Some(filmcraft_ui_egui::state::TRACK_H_MAX as f64));
}
