//! Playback clock: the audio device drives the playhead while it plays, and a device that opens
//! but never consumes samples (seen on Linux with ALSA, #136) must not freeze playback.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_time::Tick;
use filmcraft_ui_egui::{AudioOut, FilmcraftApp};
use serde_json::json;

/// An output whose stream opens, then plays `per_poll` frames each time the clock is read
/// (0 = a stalled device that never calls back).
struct FakeOut {
    played: Arc<AtomicU64>,
    per_poll: u64,
    started: bool,
}

impl AudioOut for FakeOut {
    fn start(&mut self, _fill: Box<dyn FnMut(&mut [f32], usize) + Send>) -> Result<u32, String> {
        self.played.store(0, Ordering::SeqCst);
        self.started = true;
        Ok(48_000)
    }
    fn stop(&mut self) {
        self.started = false;
    }
    fn sample_rate(&self) -> u32 {
        48_000
    }
    fn played_frames(&self) -> Option<u64> {
        self.started.then(|| self.played.fetch_add(self.per_poll, Ordering::SeqCst) + self.per_poll)
    }
}

/// Play for `steps` × 0.25 s of simulated time with the given output; the playhead in seconds.
fn play_with(per_poll: u64, steps: usize) -> (f64, bool) {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.set_playhead(Tick::ZERO);
    let mut app = FilmcraftApp::new(s);
    app.audio = Some(Box::new(FakeOut { played: Arc::new(AtomicU64::new(0)), per_poll, started: false }));
    let mut h = Harness::builder().with_size(egui::vec2(1280.0, 800.0)).with_step_dt(0.25).build_eframe(move |_cc| app);
    h.step();
    h.state_mut().play(1.0);
    for _ in 0..steps {
        h.step();
    }
    let app = h.state_mut();
    (app.session.playhead().seconds(), app.playback.audio_clock)
}

/// #136: the stream opened, the device never asked for samples, the audio clock stayed at 0 and
/// the playhead never moved. After a short grace period playback falls back to the wall clock.
#[test]
fn a_stalled_audio_device_does_not_freeze_playback() {
    let (t, audio_clock) = play_with(0, 16);
    assert!(t > 1.5, "the playhead should move with the wall clock, it is at {t:.2} s");
    assert!(!audio_clock, "the stalled device no longer drives the clock");
}

/// Control: a device that plays keeps driving the clock (0.1 s of audio per poll).
#[test]
fn a_playing_audio_device_drives_the_clock() {
    let (t, audio_clock) = play_with(4_800, 8);
    assert!(t > 0.2, "{t:.2} s");
    assert!(audio_clock);
}

/// One frame, with the app's raw-input hook (where the keyboard shortcuts are read) as eframe runs it.
fn frame(h: &mut Harness<'static, FilmcraftApp>) {
    let ctx = h.ctx.clone();
    let mut raw = std::mem::take(h.input_mut());
    eframe::App::raw_input_hook(h.state_mut(), &ctx, &mut raw);
    *h.input_mut() = raw;
    h.step();
}

fn press(h: &mut Harness<'static, FilmcraftApp>, key: egui::Key, shift: bool) {
    let modifiers = egui::Modifiers { shift, ..Default::default() };
    h.input_mut().events.push(egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers });
    h.input_mut().events.push(egui::Event::Key { key, physical_key: None, pressed: false, repeat: false, modifiers });
    frame(h);
}

/// Premiere: Up / Down Arrow (Go to Previous / Next Edit Point) work while playing: the playhead
/// jumps to the cut and playback carries on from there, without pausing. FilmCraft ran the
/// command, then the playback clock put the playhead straight back.
#[test]
fn up_and_down_jump_to_edits_while_playing_and_keep_playing() {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.set_playhead(Tick::ZERO);
    let edits: Vec<Tick> = s.active_sequence().unwrap().edit_points();
    let mut app = FilmcraftApp::new(s);
    app.audio = Some(Box::new(FakeOut { played: Arc::new(AtomicU64::new(0)), per_poll: 1_200, started: false }));
    let mut h = Harness::builder().with_size(egui::vec2(1280.0, 800.0)).with_step_dt(0.025).build_eframe(move |_cc| app);
    frame(&mut h);
    h.state_mut().play(1.0);
    // through the preroll (the first frames), then a little way in
    for _ in 0..400 {
        frame(&mut h);
        if h.state().session.playhead() > Tick::from_seconds_f64(0.2) {
            break;
        }
    }
    let p0 = h.state().session.playhead();
    assert!(h.state().playback.playing && p0 > Tick::ZERO, "playing: {p0:?}");
    let next_after = |t: Tick| edits.iter().copied().filter(|e| *e > t).min();
    let near = |got: Tick, want: Tick| got >= want && got.seconds() - want.seconds() < 0.2;

    // Down: the next edit point, still playing, and playing on from there
    press(&mut h, egui::Key::ArrowDown, true);
    let p1 = h.state().session.playhead();
    let e1 = next_after(p0).expect("the demo has edits after the start");
    assert!(near(p1, e1), "Shift+Down from {:.3} s went to {:.3} s, the next edit is {:.3} s", p0.seconds(), p1.seconds(), e1.seconds());
    assert!(h.state().playback.playing, "still playing");
    for _ in 0..4 {
        frame(&mut h);
    }
    let p2 = h.state().session.playhead();
    assert!(p2 > p1 && p2.seconds() - e1.seconds() < 0.6, "playback carried on from the edit: {:.3} s", p2.seconds());
    assert!(h.state().playback.playing);

    // Up: back to the edit before, still playing. The cut just played over (`e1`, a few frames
    // behind the playhead) is passed over, as in Premiere: Up goes to the one before it.
    let prev_before = |t: Tick| edits.iter().copied().filter(|e| *e < t).max();
    let p3 = h.state().session.playhead();
    assert_eq!(prev_before(p3), Some(e1));
    press(&mut h, egui::Key::ArrowUp, true);
    let p4 = h.state().session.playhead();
    let e0 = prev_before(e1).unwrap_or(Tick::ZERO);
    assert!(near(p4, e0), "Shift+Up from {:.3} s went to {:.3} s, the edit before is {:.3} s", p3.seconds(), p4.seconds(), e0.seconds());
    assert!(h.state().playback.playing);

    // plain Down (targeted tracks) moves forward too, while playing
    let p5 = h.state().session.playhead();
    press(&mut h, egui::Key::ArrowDown, false);
    let p6 = h.state().session.playhead();
    assert!(p6.seconds() > p5.seconds() + 0.1 && edits.iter().any(|e| near(p6, *e)), "Down while playing: {:.3} → {:.3} s", p5.seconds(), p6.seconds());
    assert!(h.state().playback.playing);
    h.state_mut().stop();
}

/// Premiere: Up pressed again and again while playing steps back cut by cut. Each press used to
/// land on the cut just behind the playhead, which after the first press is the cut it had just
/// gone to (playback carries the playhead a few frames past it before the next press).
#[test]
fn up_repeated_while_playing_steps_back_cut_by_cut() {
    for shift in [false, true] {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let rate = s.sequence_rate();
        // plain Up follows the targeted tracks: target V1 only
        let v1 = s.active_sequence().unwrap().video_tracks[0].id;
        for t in s.targeting().targeted {
            s.execute("timeline.setTargeting", json!({"track": t.0, "targeted": t == v1})).unwrap();
        }
        s.execute("timeline.setTargeting", json!({"track": v1.0, "targeted": true})).unwrap();
        let seq = s.active_sequence().unwrap();
        let mut edits: Vec<Tick> = if shift { seq.edit_points() } else { seq.video_tracks[0].items.iter().flat_map(|i| [i.start, i.end()]).collect() };
        edits = edits.into_iter().map(|e| rate.snap_edit(e)).collect();
        edits.sort();
        edits.dedup();
        assert!(edits.len() >= 4, "{edits:?}");
        let (c1, c2, c3) = (edits[0], edits[1], edits[2]);
        // mid-clip in clip 3
        s.set_playhead(Tick((c3.0 + edits[3].0) / 2));
        let mut app = FilmcraftApp::new(s);
        app.audio = Some(Box::new(FakeOut { played: Arc::new(AtomicU64::new(0)), per_poll: 1_200, started: false }));
        let mut h = Harness::builder().with_size(egui::vec2(1280.0, 800.0)).with_step_dt(0.025).build_eframe(move |_cc| app);
        frame(&mut h);
        h.state_mut().play(1.0);
        let start = h.state().session.playhead();
        for _ in 0..400 {
            frame(&mut h);
            if h.state().session.playhead() > start {
                break;
            }
        }
        assert!(h.state().playback.playing && h.state().session.playhead() < edits[3], "playing in clip 3");
        for want in [c3, c2, c1] {
            press(&mut h, egui::Key::ArrowUp, shift);
            let got = h.state().session.playhead();
            assert_eq!(got, want, "Up (shift {shift}) went to {:.3} s, want {:.3} s", got.seconds(), want.seconds());
            assert!(h.state().playback.playing, "still playing");
            // playback carries on a few frames from the cut before the next press
            for _ in 0..4 {
                frame(&mut h);
            }
            let on = h.state().session.playhead();
            assert!(on > want && on.seconds() - want.seconds() < 0.5, "played on from the cut: {:.3} s", on.seconds());
        }
        assert!(h.state().playback.playing);
        h.state_mut().stop();
    }
}

/// A demo-project app in a harness with a working audio output, `return_to_beginning` as given.
fn demo_app(return_to_beginning: bool) -> Harness<'static, FilmcraftApp> {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.prefs.timeline.return_to_beginning = return_to_beginning;
    let mut app = FilmcraftApp::new(s);
    app.audio = Some(Box::new(FakeOut { played: Arc::new(AtomicU64::new(0)), per_poll: 1_200, started: false }));
    let mut h = Harness::builder().with_size(egui::vec2(1280.0, 800.0)).with_step_dt(0.025).build_eframe(move |_cc| app);
    frame(&mut h);
    h
}

/// Premiere: no wrapping at the ends of the timeline while playing. Down past the last edit goes
/// to the sequence end, where playback stops (it does not start again from the beginning); Up at
/// the start stays at the start and plays on from there.
#[test]
fn edit_keys_while_playing_do_not_wrap_at_the_ends() {
    for shift in [false, true] {
        let mut h = demo_app(false);
        let end = {
            let s = &h.state().session;
            s.sequence_rate().snap_edit(s.active_sequence().unwrap().duration())
        };
        h.state_mut().session.set_playhead(Tick::ZERO);
        h.state_mut().play(1.0);
        for _ in 0..400 {
            frame(&mut h);
            if h.state().session.playhead() > Tick::ZERO {
                break;
            }
        }
        // Down until playback reaches the end: always forward, ends at the sequence end, stopped
        let mut last = h.state().session.playhead();
        for _ in 0..60 {
            press(&mut h, egui::Key::ArrowDown, shift);
            frame(&mut h);
            let t = h.state().session.playhead();
            assert!(t >= last, "Down (shift {shift}) went back from {:.3} s to {:.3} s", last.seconds(), t.seconds());
            last = t;
        }
        assert_eq!(h.state().session.playhead(), end, "Down (shift {shift}) ends at the sequence end");
        assert!(!h.state().playback.playing, "playback stopped at the end");
        for _ in 0..20 {
            frame(&mut h);
        }
        assert_eq!(h.state().session.playhead(), end, "and stays there");

        // Up from the start while playing: stays at the start (never jumps to the end), plays on
        h.state_mut().session.set_playhead(Tick::ZERO);
        h.state_mut().play(1.0);
        for _ in 0..400 {
            frame(&mut h);
            if h.state().session.playhead() > Tick::ZERO {
                break;
            }
        }
        for _ in 0..5 {
            press(&mut h, egui::Key::ArrowUp, shift);
            assert_eq!(h.state().session.playhead(), Tick::ZERO, "Up (shift {shift}) near the start goes to the start");
            frame(&mut h);
            assert!(h.state().session.playhead().seconds() < 0.5, "playing on from the start");
        }
        assert!(h.state().playback.playing);
        h.state_mut().stop();
    }
}

/// Playback that reaches the end of the sequence stops on its last edge, with Loop off. The
/// "return to beginning" preference only matters when playback is started again from there: off,
/// it plays nothing and stays; on, it starts from the beginning (Premiere).
#[test]
fn playback_stops_at_the_sequence_end_without_wrapping() {
    for rtb in [false, true] {
        let mut h = demo_app(rtb);
        let (end, near_end) = {
            let s = &h.state().session;
            let rate = s.sequence_rate();
            let end = rate.snap_edit(s.active_sequence().unwrap().duration());
            (end, Tick(end.0.saturating_sub(rate.tick_of(6).0)))
        };
        assert!(!h.state().playback.looping);
        h.state_mut().session.set_playhead(near_end);
        h.state_mut().play(1.0);
        for _ in 0..400 {
            frame(&mut h);
            if !h.state().playback.playing {
                break;
            }
        }
        assert!(!h.state().playback.playing, "stopped at the end (return to beginning {rtb})");
        for _ in 0..20 {
            frame(&mut h);
        }
        assert_eq!(h.state().session.playhead(), end, "stays on the end, never wraps to 0 (return to beginning {rtb})");
        // play again from the end
        h.state_mut().play(1.0);
        for _ in 0..40 {
            frame(&mut h);
        }
        let t = h.state().session.playhead();
        if rtb {
            assert!(t < end && t.seconds() < 2.0, "restarts from the beginning: {:.3} s", t.seconds());
        } else {
            assert_eq!(t, end, "return to beginning off: stays at the end");
            assert!(!h.state().playback.playing);
        }
        h.state_mut().stop();
    }
}
