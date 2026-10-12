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

    // Up: back to the edit before, still playing
    let prev_before = |t: Tick| edits.iter().copied().filter(|e| *e < t).max();
    let p3 = h.state().session.playhead();
    press(&mut h, egui::Key::ArrowUp, true);
    let p4 = h.state().session.playhead();
    let e0 = prev_before(p3).unwrap_or(Tick::ZERO);
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
