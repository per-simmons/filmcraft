use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::Session;
use crate::media_test_util::tmp_dir;
use filmcraft_project::{ItemId, SequenceSettings, TrackKind};
use filmcraft_time::{FrameRate, Tick, TimeRange};

const SR: u32 = 48_000;
/// The three "words" of the fixture (seconds): tone bursts with quiet noise between them.
const WORDS: [(&str, f64, f64); 3] = [("alpha", 0.4, 1.0), ("bravo", 1.8, 2.4), ("charlie", 3.0, 3.6)];
const LEN: f64 = 4.2;

/// A mono 16-bit WAV: −66 dBFS noise, with a voiced 180 Hz burst (10 ms ramps) for each word.
fn write_wav(path: &Path) {
    let n = (LEN * f64::from(SR)) as usize;
    let mut seed = 12345u32;
    let mut pcm = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f64 / f64::from(SR);
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let mut v = ((seed >> 16) as f64 / 32768.0 - 1.0) * 0.0005;
        for (_, a, b) in WORDS {
            if t >= a && t < b {
                let ramp = ((t - a).min(b - t) / 0.01).min(1.0);
                let ph = 2.0 * std::f64::consts::PI * 180.0 * t;
                v += ramp * 0.25 * (ph.sin() + 0.3 * (3.0 * ph).sin());
            }
        }
        pcm.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    let mut w = Vec::new();
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&SR.to_le_bytes());
    w.extend_from_slice(&(SR * 2).to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    w.extend_from_slice(&pcm);
    std::fs::write(path, w).unwrap();
}

/// The fixture on A1 of a 25 fps sequence, transcribed (the three words) with its voice measured
/// from the waveform.
fn session() -> (Session, ItemId) {
    let dir = tmp_dir("waveform");
    let path = dir.join("voice.wav");
    write_wav(&path);
    let mut s = Session::default();
    let r = s.execute("file.import", json!({"paths": [path.to_string_lossy()]})).unwrap();
    let item = ItemId(r["items"][0].as_u64().unwrap());
    let rate = FrameRate::new(25, 1);
    let mut p = (*s.project).clone();
    let seq = p.new_sequence("Seq", SequenceSettings { width: 640, height: 360, frame_rate: rate, ..Default::default() }, 1, 1, None);
    let d = p.item(item).unwrap().duration();
    let a = p.make_track_item(item, TrackKind::Audio, Tick::ZERO, TimeRange::new(Tick::ZERO, d), rate).unwrap();
    p.sequence_mut(seq).unwrap().audio_tracks[0].items.push(a);
    s.project = Arc::new(p);
    s.state.active_sequence = Some(seq);
    s.state.open_sequences = vec![seq];
    let words: Vec<Value> = WORDS
        .iter()
        .map(|(t, a, b)| json!({"text": t, "start": Tick::from_seconds_f64(*a).0, "end": Tick::from_seconds_f64(*b).0, "confidence": 1.0}))
        .collect();
    s.execute("transcript.set", json!({"item": item.0, "transcript": {"language": "en", "words": words}})).unwrap();
    s.execute("transcript.findPauses", json!({"items": [item.0]})).unwrap();
    assert!(!s.project.transcripts[&item].voice.is_empty(), "the voice map was measured");
    (s, item)
}

fn cut(s: &mut Session, a: f64, b: f64) {
    let r = TimeRange::from_bounds(Tick::from_seconds_f64(a), Tick::from_seconds_f64(b));
    s.edit_sequence("cut", |q, ctx, _| {
        let tracks: Vec<_> = q.all_tracks().map(|t| t.id).collect();
        filmcraft_edit::extract(q, &tracks, r, ctx);
        Ok(())
    })
    .unwrap();
}

fn near(v: &Value, want: f64, tol: f64) -> bool {
    v.as_f64().is_some_and(|x| (x - want).abs() <= tol)
}

fn issues(r: &Value) -> Vec<String> {
    let mut v: Vec<String> =
        r["list"].as_array().unwrap().iter().flat_map(|j| j["issues"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string())).collect();
    v.sort();
    v
}

#[test]
fn silences_measure_the_dead_air_of_the_mix() {
    let (mut s, _) = session();
    let r = s.execute("waveform.silences", json!({})).unwrap();
    assert_eq!(r["count"], 4, "{r}");
    let runs = r["runs"].as_array().unwrap();
    for (run, (a, b)) in runs.iter().zip([(0.0, 0.4), (1.0, 1.8), (2.4, 3.0), (3.6, 4.2)]) {
        assert!(near(&run["start"], a, 0.02) && near(&run["end"], b, 0.02), "{run}");
    }
    assert_eq!(runs[0]["kind"], "head");
    assert_eq!(runs[1]["after"], "alpha");
    assert_eq!(runs[1]["before"], "bravo");
    assert_eq!(runs[3]["kind"], "tail");
    assert!(near(&r["totalSeconds"], 2.4, 0.06), "{r}");
    let r = s.execute("waveform.silences", json!({"minSeconds": 0.7})).unwrap();
    assert_eq!(r["count"], 1, "only the 0.8 s gap is that long: {r}");
}

#[test]
fn envelope_reads_levels_voice_and_cuts() {
    let (mut s, _) = session();
    cut(&mut s, 1.2, 1.6);
    let r = s.execute("waveform.envelope", json!({"from": 0.9, "to": 1.5, "windowMs": 10})).unwrap();
    let rows = r["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 60, "{}", rows.len());
    let at = |sec: f64| rows.iter().find(|x| near(&x[0], sec, 0.004)).unwrap();
    assert!(at(0.95)[1].as_f64().unwrap() > -20.0, "inside alpha: {}", at(0.95));
    assert_eq!(at(0.95)[3], 1, "voiced");
    assert!(at(1.1)[1].as_f64().unwrap() < -55.0, "the pause: {}", at(1.1));
    assert_eq!(at(1.1)[3], 0);
    assert_eq!(at(1.2)[4], 1, "the cut is marked");
    assert!(at(1.45)[1].as_f64().unwrap() > -20.0, "bravo now starts 0.4 s earlier: {}", at(1.45));
    assert_eq!(r["strip"]["cuts"].as_str().unwrap().chars().filter(|c| *c == '|').count(), 1);
    assert_eq!(r["words"].as_array().unwrap().len(), 2, "alpha and bravo: {}", r["words"]);
    assert!(s.execute("waveform.envelope", json!({"around": 1.0, "span": 0.2})).unwrap()["rows"].as_array().unwrap().len() >= 20);
}

#[test]
fn joins_catch_clipped_words_and_fix_restores_them() {
    let (mut s, _) = session();
    // through the end of "bravo" and the start of "charlie"
    cut(&mut s, 2.0, 3.2);
    let r = s.execute("waveform.joins", json!({"tighten": false})).unwrap();
    let mid = r["list"].as_array().unwrap().iter().find(|j| near(&j["seconds"], 2.0, 0.001)).unwrap().clone();
    assert_eq!(mid["tail"]["clipsWord"]["text"], "bravo", "{mid}");
    assert!(near(&mid["tail"]["clipsWord"]["lostMs"], 400.0, 1.0), "{mid}");
    assert_eq!(mid["head"]["clipsWord"]["text"], "charlie", "{mid}");
    assert!(near(&mid["head"]["clipsWord"]["lostMs"], 200.0, 1.0), "{mid}");
    assert_eq!(mid["issues"], json!(["clipsWordTail", "clipsWordOnset"]));
    // tail: to 2.4 + 30 ms, up to whole frames (0.44 s); head: back to 3.0 − 35 ms (0.24 s)
    assert!(near(&mid["fix"]["tailMs"], 440.0, 0.5) && near(&mid["fix"]["headMs"], 240.0, 0.5), "{mid}");

    let before = s.active_sequence().unwrap().duration();
    let f = s.execute("waveform.fixJoins", json!({"joins": [mid["join"]]})).unwrap();
    assert_eq!(f["applied"], 1, "{f}");
    let grown = s.active_sequence().unwrap().duration() - before;
    assert!((grown.seconds() - 0.68).abs() < 0.001, "both words back: {}", grown.seconds());
    let r = s.execute("waveform.joins", json!({"tighten": false})).unwrap();
    assert!(!issues(&r).iter().any(|i| i.starts_with("clipsWord")), "{r}");

    // one undo step
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(s.active_sequence().unwrap().duration(), before);
}

#[test]
fn loose_silence_at_a_cut_is_tightened_to_the_margins() {
    let (mut s, _) = session();
    cut(&mut s, 1.2, 1.6);
    let r = s.execute("waveform.joins", json!({})).unwrap();
    let mid = r["list"].as_array().unwrap().iter().find(|j| near(&j["seconds"], 1.2, 0.001)).unwrap().clone();
    assert_eq!(mid["issues"], json!(["looseTail", "looseHead"]), "{mid}");
    assert!(near(&mid["quietAcrossMs"], 400.0, 20.0), "{mid}");
    assert!(
        near(&mid["fix"]["tailMs"], -160.0, 0.5) && near(&mid["fix"]["headMs"], -120.0, 0.5),
        "voice edges sit 7.5 ms out: 192.5 − 35 ms rounds down to 3 frames: {mid}"
    );
    // the sequence head (0.4 s before alpha) and tail (0.6 s after charlie) are loose too
    assert_eq!(issues(&r), ["looseHead", "looseHead", "looseTail", "looseTail"]);

    let dry = s.execute("waveform.fixJoins", json!({"dryRun": true})).unwrap();
    assert_eq!(dry["planned"].as_array().unwrap().len(), 3, "{dry}");
    assert_eq!(dry["applied"], 0);
    s.execute("waveform.fixJoins", json!({})).unwrap();
    let r = s.execute("waveform.joins", json!({})).unwrap();
    assert!(issues(&r).is_empty(), "{r}");
    // no dead air of 150 ms or more is left anywhere, and every word survives
    let d = s.execute("waveform.silences", json!({})).unwrap();
    assert!(d["runs"].as_array().unwrap().iter().all(|x| x["kind"] == "inClip"), "only the untouched pauses inside the clip: {d}");
    let w = s.execute("transcript.inspect", json!({})).unwrap();
    assert_eq!(w["words"].as_array().unwrap().len(), 3);
}

#[test]
fn untranscribed_and_empty_sequences_never_fail_badly() {
    let mut s = Session::default();
    assert!(s.execute("waveform.joins", json!({})).is_err(), "no sequence: disabled");
    let (mut s, item) = session();
    s.execute("transcript.delete", json!({"items": [item.0]})).unwrap();
    let r = s.execute("waveform.joins", json!({"all": true})).unwrap();
    assert!(issues(&r).iter().all(|i| i == "noVoiceMap"), "{r}");
    assert_eq!(s.execute("waveform.fixJoins", json!({})).unwrap()["applied"], 0);
    assert!(s.execute("waveform.envelope", json!({"from": 3.0, "to": 2.0})).is_err());
    assert!(s.execute("waveform.envelope", json!({"from": 0, "to": 4.2, "windowMs": 5, "points": 10})).is_ok());
}

#[test]
fn loudness_measures_the_mix_against_pats_target() {
    let (mut s, _) = session();
    let r = s.execute("waveform.loudness", json!({"perTrack": true})).unwrap();
    // the bursts peak at 0.25 × (1 + 0.3) ≈ −9.7 dBFS; the mono clip reaches each side of the
    // stereo mix 3 dB down (pan law)
    assert!(near(&r["samplePeakDbfs"], -12.7, 0.6), "{r}");
    assert!(r["truePeakDbtp"].as_f64().unwrap() >= r["samplePeakDbfs"].as_f64().unwrap() - 0.01, "{r}");
    let lufs = r["integratedLufs"].as_f64().unwrap();
    assert!((-25.0..-8.0).contains(&lufs), "{r}");
    assert_eq!(r["measured"], "mix");
    assert!(near(&r["target"]["gainDb"], -14.0 - lufs, 0.02), "{r}");
    assert_eq!(r["perTrack"].as_array().unwrap().len(), 1, "only A1 has clips: {r}");
    assert!(near(&r["perTrack"][0]["integratedLufs"], lufs, 0.05), "{r}");
    // a track with nothing on it is silent: no loudness, no target
    let r = s.execute("waveform.loudness", json!({"tracks": [7]})).unwrap();
    assert!(r["integratedLufs"].is_null() && r.get("target").is_none(), "{r}");
    let r = s.execute("waveform.loudness", json!({"from": 1.8, "to": 2.4})).unwrap();
    assert!(r["integratedLufs"].as_f64().unwrap() > lufs, "one word alone is louder than the whole with its pauses: {r}");
}

#[test]
fn a_tight_cut_keeps_the_original_noise_floor_as_the_ruler() {
    // With the silence cut out, the quietest tenth of the mix is speech: a floor taken from all of
    // it put the gate at −35 dBFS, on top of word tails (live take 1.9: 150 ms "dead air" that was
    // a decaying vowel). The floor comes from where the voice map has no voice.
    let (mut s, _) = session();
    cut(&mut s, 2.45, 2.95);
    cut(&mut s, 1.05, 1.75);
    s.execute("waveform.fixJoins", json!({})).unwrap();
    let r = s.execute("waveform.silences", json!({"minSeconds": 0.05})).unwrap();
    assert_eq!(r["floorFrom"], "outsideVoice", "{r}");
    assert!(r["floorDb"].as_f64().unwrap() < -60.0 && r["gateDb"] == -50.0, "{r}");
    // what is left at each cut is the 30 + 35 ms margins, rounded to whole 40 ms frames
    assert!(r["runs"].as_array().unwrap().iter().all(|x| x["ms"].as_f64().unwrap() <= 110.0), "{r}");
}

// ---------------------------------------------------------------------------------------------
// Slivers: text edits never leave a 1–2-frame piece of a clip at a seam
// ---------------------------------------------------------------------------------------------

fn sec(x: f64) -> Tick {
    Tick::from_seconds_f64(x)
}

/// The fixture's media as `takes` (timeline start, media In, media Out) on A1 of the 25 fps
/// sequence, transcribed as `words` (media seconds) with their spans as the voice map.
fn takes_session(takes: &[(f64, f64, f64)], words: &[(&str, f64, f64)]) -> Session {
    let (mut s, item) = session();
    let rate = FrameRate::new(25, 1);
    let seq = s.state.active_sequence.unwrap();
    let mut p = (*s.project).clone();
    let items: Vec<_> =
        takes.iter().map(|(at, a, b)| p.make_track_item(item, TrackKind::Audio, sec(*at), TimeRange::from_bounds(sec(*a), sec(*b)), rate).unwrap()).collect();
    p.sequence_mut(seq).unwrap().audio_tracks[0].items = items;
    s.project = Arc::new(p);
    let w: Vec<Value> = words.iter().map(|(t, a, b)| json!({"text": t, "start": sec(*a).0, "end": sec(*b).0, "confidence": 1.0})).collect();
    let v: Vec<Value> = words.iter().map(|(_, a, b)| json!([sec(*a).0, sec(*b).0])).collect();
    s.execute("transcript.set", json!({"item": item.0, "transcript": {"language": "en", "words": w, "voice": v}})).unwrap();
    s.active_sequence().unwrap().check().unwrap();
    s
}

/// Every clip on every track is at least three frames long, and `waveform.joins` finds no cut
/// through a word and no voice running across a cut.
fn assert_no_slivers(s: &mut Session) {
    let q = s.active_sequence().unwrap();
    let fd = q.settings.frame_rate.frame_duration();
    for t in q.all_tracks() {
        for i in &t.items {
            assert!(i.duration >= Tick(fd.0 * 3), "a {}-frame sliver at {:.3} s: {:?}", i.duration.0 as f64 / fd.0 as f64, i.start.seconds(), t.items);
        }
    }
    let r = s.execute("waveform.joins", json!({"tighten": false})).unwrap();
    assert!(!issues(&r).iter().any(|i| i.starts_with("clipsWord") || i.starts_with("voiceRuns")), "{r}");
}

fn word_texts(s: &mut Session) -> Vec<String> {
    let r = s.execute("transcript.inspect", json!({})).unwrap();
    r["words"].as_array().unwrap().iter().map(|w| w["text"].as_str().unwrap().to_string()).collect()
}

const THREE: [(&str, f64, f64); 3] = [("alpha", 0.40, 1.00), ("bravo", 1.80, 2.40), ("charlie", 3.00, 3.60)];

/// Live 2026-10-12: the lead-in pause deleted (keeping 35 ms before the first word), then the
/// first word extracted: the cut started on the frame the word starts in and left the frame
/// before it, 40 ms of silence, as a clip of its own at the sequence head.
#[test]
fn extracting_the_first_words_leaves_no_sliver_at_the_head() {
    let mut s = takes_session(&[(0.0, 0.0, 4.2)], &THREE);
    let r = s.execute("transcript.deleteHits", json!({"filter": "pauses", "hit": 0})).unwrap();
    assert!(near(&r["seconds"], 0.36, 0.001), "the lead-in less the 35 ms margin, down to frames: {r}");
    let r = s.execute("transcript.extract", json!({"from": 0})).unwrap();
    assert_eq!(r["start"], 0, "the cut runs from the sequence head: {r}");
    assert_eq!(word_texts(&mut s), ["bravo", "charlie"]);
    assert_no_slivers(&mut s);
}

/// Live 2026-10-12: take 1.1 appended after take 1.0, each trimmed a couple of frames from its
/// words. Extracting the first word of 1.1 left 1.1's two frames of lead-in before the cut, and
/// extracting the last word of 1.0 left 1.0's two frames of tail after it.
#[test]
fn extracting_words_at_a_clip_boundary_leaves_no_sliver() {
    // take 1.0 ends 80 ms after "alpha"; take 1.1 starts 80 ms before "bravo"
    let mut s = takes_session(&[(0.0, 0.0, 1.08), (1.08, 1.72, 4.2)], &THREE);
    let r = s.execute("transcript.extract", json!({"from": 1})).unwrap();
    assert_eq!(r["start"].as_i64().unwrap(), sec(1.08).0, "snapped back to the take's In point: {r}");
    assert_no_slivers(&mut s);
    let r = s.execute("transcript.extract", json!({"from": 0})).unwrap();
    assert_eq!(r["end"].as_i64().unwrap(), sec(1.08).0, "snapped on to the take's Out point: {r}");
    assert_eq!(word_texts(&mut s), ["charlie"]);
    assert_no_slivers(&mut s);
    // nothing of the words around the cuts went: the lead-in and "charlie" are whole
    let q = s.active_sequence().unwrap();
    assert_eq!(q.duration(), sec(3.56 - 1.36), "two 0.68 s cuts from 1.08 + 2.48 s of takes");
}

/// Live 2026-10-12 ("about | What"): two fillers 40 ms apart each rounded to their nearest frames
/// left the frame between the cuts, the end of "um", as a 1-frame clip (an audible blip).
#[test]
fn adjacent_cuts_leave_no_sliver_between_them() {
    let words = [("about", 0.40, 0.90), ("um", 1.00, 1.30), ("uh", 1.34, 1.60), ("What", 1.80, 2.40), ("now", 3.00, 3.60)];
    let mut s = takes_session(&[(0.0, 0.0, 4.2)], &words);
    let before = s.active_sequence().unwrap().duration();
    let r = s.execute("transcript.deleteHits", json!({"filter": "fillers"})).unwrap();
    assert_eq!(r["removed"], 2, "{r}");
    assert_eq!(word_texts(&mut s), ["about", "What", "now"]);
    assert_eq!(s.active_sequence().unwrap().duration(), before - sec(0.60), "1.00–1.60 in one cut");
    assert_no_slivers(&mut s);
}

/// Speech between two cuts stays, however short: a word, or voice with no word on it.
#[test]
fn speech_between_close_cuts_is_never_removed() {
    // "a" (40 ms) sits between the two fillers
    let words = [("about", 0.40, 0.90), ("um", 1.00, 1.28), ("a", 1.29, 1.33), ("uh", 1.34, 1.60), ("What", 1.80, 2.40)];
    let mut s = takes_session(&[(0.0, 0.0, 4.2)], &words);
    s.execute("transcript.deleteHits", json!({"filter": "fillers"})).unwrap();
    assert_eq!(word_texts(&mut s), ["about", "a", "What"]);
}

#[test]
fn cut_silences_cuts_every_quiet_run_down_to_the_margins() {
    // The voice map counts a word's decaying tail as voice, so pause cuts made from it leave
    // ~120–140 ms at every cut (Muse Wonder 1.0, 2026-10-12). This pass measures the mix itself.
    let (mut s, _) = session();
    let dry = s.execute("waveform.cutSilences", json!({"dryRun": true})).unwrap();
    assert_eq!(dry["cuts"].as_array().unwrap().len(), 4, "head, two gaps, tail: {dry}");
    let r = s.execute("waveform.cutSilences", json!({})).unwrap();
    assert!(near(&r["durationAfter"], 1.8 + 0.13 * 2.0, 0.1), "three words plus margins: {r}");
    let left = s.execute("waveform.silences", json!({"minSeconds": 0.1})).unwrap();
    assert_eq!(left["count"], 0, "nothing of 100 ms or more is left: {left}");
    assert_eq!(s.execute("transcript.inspect", json!({})).unwrap()["words"].as_array().unwrap().len(), 3);
    // one undo step
    s.execute("edit.undo", json!({})).unwrap();
    assert!(near(&json!(s.active_sequence().unwrap().duration().seconds()), LEN, 0.01));
}
