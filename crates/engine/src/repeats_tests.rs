use serde_json::{Value, json};

use crate::Session;
use crate::repeats::{Repeat, RepeatOptions, find_repeats, normalize};
use filmcraft_edit::transcript::SeqWord;
use filmcraft_project::{ClipId, ItemId};
use filmcraft_time::Tick;

/// Words spoken one after another, 0.3 s each with 0.05 s between them. `|` is a 1 s pause, `||`
/// a 1 s pause and a new clip (another take).
fn spoken(script: &str) -> Vec<SeqWord> {
    let mut t = 0.0;
    let mut clip = 1;
    let mut out = Vec::new();
    for tok in script.split_whitespace() {
        match tok {
            "|" => t += 1.0,
            "||" => {
                t += 1.0;
                clip += 1;
            }
            w => {
                out.push(SeqWord {
                    text: w.into(),
                    start: Tick::from_seconds_f64(t),
                    end: Tick::from_seconds_f64(t + 0.3),
                    clip: ClipId(clip),
                    item: ItemId(clip),
                    index: out.len(),
                    track: 0,
                    speaker: None,
                    confidence: 1.0,
                });
                t += 0.35;
            }
        }
    }
    out
}

fn find(script: &str) -> (Vec<SeqWord>, Vec<Repeat>) {
    let w = spoken(script);
    let h = find_repeats(&w, &RepeatOptions::default());
    (w, h)
}

/// The words a cut removes, joined.
fn cut_text(w: &[SeqWord], h: &Repeat) -> String {
    w[h.cut.clone()].iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
}

#[test]
fn normalises_case_punctuation_and_curly_apostrophes() {
    assert_eq!(normalize("So,"), "so");
    assert_eq!(normalize("It\u{2019}s"), "it's");
    assert_eq!(normalize("downloaded…"), "downloaded");
    assert_eq!(normalize("“Anyway"), "anyway");
}

#[test]
fn a_restart_cuts_the_abandoned_attempt() {
    // Pat's take: both readings stayed in a REVIEW
    let (w, h) = find(
        "This app is everywhere. | So it's not only highly downloaded… | um | So it's not only highly downloaded, being one of the top applications on the store.",
    );
    assert_eq!(h.len(), 1, "{h:?}");
    assert!(!h[0].near);
    assert_eq!(h[0].matched, 6);
    assert_eq!(cut_text(&w, &h[0]), "So it's not only highly downloaded… um");
    assert_eq!(w[h[0].later].text, "So", "the later reading starts at its first word");
    assert_eq!(h[0].later, h[0].cut.end, "the cut stops right before the later reading");
    assert!(!h[0].complete);
}

#[test]
fn a_near_identical_restart_is_found() {
    let (w, h) = find("And this is the camera | which I've reviewed in the past, | which I've done in the past, and it held up really well.");
    assert_eq!(h.len(), 1, "{h:?}");
    assert!(h[0].near);
    assert_eq!(cut_text(&w, &h[0]), "which I've reviewed in the past,");
    // a dropped word too
    let (w, h) = find("Anyway, so let's look at the pricing | Anyway, let's look at the pricing page for a second.");
    assert_eq!(h.len(), 1, "{h:?}");
    assert!(h[0].near);
    assert_eq!(cut_text(&w, &h[0]), "Anyway, so let's look at the pricing");
}

#[test]
fn a_complete_reading_read_again_still_cuts_the_earlier() {
    // "Anyway… / Anyway…": the first transition was complete; the last complete reading is kept
    let (w, h) = find(
        "That was the setup. | Anyway, let's move on to the pricing tiers. | Anyway, let's move on to the pricing tiers. | The free plan gives you ten credits.",
    );
    assert_eq!(h.len(), 1, "{h:?}");
    assert!(h[0].complete);
    assert_eq!(cut_text(&w, &h[0]), "Anyway, let's move on to the pricing tiers.");
    assert_eq!(w[h[0].later].text, "Anyway,");
    assert!(w[h[0].later].start > w[h[0].cut.start].start);
}

#[test]
fn three_attempts_give_two_adjacent_cuts() {
    let (w, h) = find("The export dialog has | the export dialog has three | The export dialog has three tabs you need.");
    assert_eq!(h.len(), 2, "{h:?}");
    assert_eq!(h[0].cut.end, h[1].cut.start, "the second cut starts where the first stopped");
    assert_eq!(cut_text(&w, &h[0]), "The export dialog has");
    assert_eq!(cut_text(&w, &h[1]), "the export dialog has three");
    assert_eq!(w[h[1].later].text, "The");
}

#[test]
fn deliberate_emphasis_is_not_a_retake() {
    let (_, h) = find("I hear you. We get it. We get it. | It keeps going and going and going and going.");
    assert!(h.is_empty(), "{h:?}");
    let w = spoken("We get it. We get it. We get it.");
    assert!(find_repeats(&w, &RepeatOptions { min_words: 2, ..Default::default() }).is_empty());
    // meaningful words, still emphasis: a phrase of up to 3 words said again at once
    let w = spoken("Ship it tonight. Ship it tonight. | This is a huge mistake. Huge mistake.");
    assert!(find_repeats(&w, &RepeatOptions { min_words: 2, ..Default::default() }).is_empty());
    // a phrase of 4 words said again is a restart
    let (_, h) = find("Never ship untested code. | um, never ship untested code to customers.");
    assert_eq!(h.len(), 1, "{h:?}");
}

#[test]
fn common_openings_and_far_reuse_are_not_retakes() {
    // stock phrasing shared by two different sentences
    let (_, h) = find("And then you can see the menu. | And then you can see the result appear here.");
    assert!(h.is_empty(), "{h:?}");
    let (_, h) = find("So we open it. | So we close it.");
    assert!(h.is_empty(), "{h:?}");
    // the same phrase used again after a lot of other speech is not a restart
    let filler = "the model writes code while we review every file it changed, then we merge anything that passes into the main \
                  branch, and after lunch somebody checks the dashboard for errors before the evening deploy goes out to customers";
    let (_, h) = find(&format!("Click the green export button. | {filler} | Click the green export button again."));
    assert!(h.is_empty(), "{h:?}");
    // beyond the window
    let w = spoken("Highly downloaded apps rank first. | Highly downloaded apps rank first.");
    let o = RepeatOptions { window: Tick::from_seconds_f64(1.0), ..Default::default() };
    assert!(find_repeats(&w, &o).is_empty());
}

#[test]
fn fillers_are_collapsed_and_a_restart_crosses_takes() {
    let (w, h) = find("This one is huge. | So it's um not only uh highly downloaded || So it's not only highly downloaded, being one of the top applications.");
    assert_eq!(h.len(), 1, "{h:?}");
    assert_ne!(w[h[0].cut.start].clip, w[h[0].later].clip, "across the clip boundary");
    assert_eq!(cut_text(&w, &h[0]), "So it's um not only uh highly downloaded");
}

#[test]
fn from_and_to_limit_the_search() {
    let script = "So it's not only highly downloaded | So it's not only highly downloaded today.";
    let w = spoken(script);
    let later = w.iter().rposition(|x| x.text == "So").unwrap();
    let o = RepeatOptions { from: Some(w[1].start), ..Default::default() };
    assert!(find_repeats(&w, &o).is_empty(), "the first attempt is outside");
    let o = RepeatOptions { to: Some(w[later].start), ..Default::default() };
    assert_eq!(find_repeats(&w, &o).len(), 1);
}

// --- commands on a real sequence ----------------------------------------------------------------

/// The demo project with two A1 clips transcribed: clip 1 (sequence 0–5 s) ends on an abandoned
/// attempt, clip 2 (5–9.5 s) is the full reading.
fn session() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let q = s.active_sequence().unwrap();
    let a = &q.audio_tracks[0].items;
    let clips = [(a[0].item, a[0].source_in, a[0].start), (a[1].item, a[1].source_in, a[1].start)];
    let takes: [(&str, f64, f64); 2] = [
        ("This app is huge. So it's not only highly downloaded…", 0.2, 0.4),
        ("So it's not only highly downloaded, being one of the top applications.", 0.2, 0.33),
    ];
    for ((item, sin, _), (text, at, step)) in clips.into_iter().zip(takes) {
        let words: Vec<Value> = text
            .split_whitespace()
            .enumerate()
            .map(|(k, w)| {
                let a = sin + Tick::from_seconds_f64(at + step * k as f64);
                json!({"text": w, "start": a.0, "end": (a + Tick::from_seconds_f64(step * 0.85)).0})
            })
            .collect();
        s.execute("transcript.set", json!({"item": item.0, "transcript": {"language": "en", "words": words}})).unwrap();
    }
    s
}

fn texts(s: &mut Session) -> Vec<String> {
    let r = s.execute("transcript.inspect", json!({})).unwrap();
    r["words"].as_array().unwrap().iter().map(|w| w["text"].as_str().unwrap().to_string()).collect()
}

#[test]
fn find_repeats_command_reports_cuts_usable_by_extract() {
    let mut s = session();
    let r = s.execute("transcript.findRepeats", json!({})).unwrap();
    let hits = r["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1, "{r}");
    let h = &hits[0];
    assert_eq!(h["kind"], "exact");
    assert_eq!(h["sharedWords"], 6);
    assert_eq!(h["earlier"]["text"], "So it's not only highly downloaded…");
    assert_eq!(h["later"]["text"], "So it's not only highly downloaded, being one of the top applications.");
    assert_eq!(h["cut"]["from"], 4);
    assert_eq!(h["cut"]["to"], 9);
    assert_eq!(h["later"]["from"], 10);
    assert!(h["cut"]["seconds"].as_f64().unwrap() > 2.0, "{h}");

    // read-only: nothing changed, nothing to undo
    assert!(s.execute("transcript.findRepeats", json!({"minWords": 7})).unwrap()["hits"].as_array().unwrap().is_empty());
    let before = s.active_sequence().unwrap().duration();

    // the cut drops straight into transcript.extract
    s.execute("transcript.extract", json!({"from": h["cut"]["from"], "to": h["cut"]["to"]})).unwrap();
    assert_eq!(texts(&mut s).join(" "), "This app is huge. So it's not only highly downloaded, being one of the top applications.");
    assert!(s.active_sequence().unwrap().duration() < before);
    s.active_sequence().unwrap().check().unwrap();
}

#[test]
fn cut_repeats_removes_every_retake_in_one_undo_step() {
    let mut s = session();
    let before = s.active_sequence().unwrap().duration();
    let dry = s.execute("transcript.cutRepeats", json!({"dryRun": true})).unwrap();
    assert_eq!(dry["removed"], 0);
    assert_eq!(dry["hits"].as_array().unwrap().len(), 1, "{dry}");
    assert_eq!(s.active_sequence().unwrap().duration(), before, "a dry run changes nothing");

    let r = s.execute("transcript.cutRepeats", json!({})).unwrap();
    assert_eq!(r["removed"], 1, "{r}");
    assert_eq!(texts(&mut s).join(" "), "This app is huge. So it's not only highly downloaded, being one of the top applications.");
    let after = s.active_sequence().unwrap().duration();
    assert_eq!(after, before - Tick(r["ticks"].as_i64().unwrap()));
    s.active_sequence().unwrap().check().unwrap();
    assert!(s.execute("transcript.findRepeats", json!({})).unwrap()["hits"].as_array().unwrap().is_empty());
    // nothing left: no edit
    assert_eq!(s.execute("transcript.cutRepeats", json!({})).unwrap()["removed"], 0);

    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(s.active_sequence().unwrap().duration(), before, "one undo step");
    assert_eq!(texts(&mut s).len(), 22);
}

#[test]
fn hostile_params_are_refused_or_clamped() {
    let mut s = session();
    for p in [
        json!({"minWords": -3}),
        json!({"minWords": "three"}),
        json!({"minWords": 1e30}),
        json!({"windowSeconds": "long"}),
        json!({"windowSeconds": -1.0}),
        json!({"from": -5}),
        json!({"to": "end"}),
    ] {
        assert!(s.execute("transcript.findRepeats", p.clone()).is_err(), "{p}");
        assert!(s.execute("transcript.cutRepeats", p.clone()).is_err(), "{p}");
    }
    for p in [
        json!({"minWords": 1}),
        json!({"windowSeconds": 0.0}),
        json!({"windowSeconds": 1e300}),
        json!({"from": 1e300}),
        json!({"from": 9, "to": 1}),
        json!(null),
    ] {
        assert!(s.execute("transcript.findRepeats", p.clone()).is_ok(), "{p}");
    }
    // no transcript: find answers empty, cut is disabled
    let mut e = Session::default();
    e.execute("file.openDemoProject", json!({})).unwrap();
    assert!(e.execute("transcript.findRepeats", json!({})).unwrap()["hits"].as_array().unwrap().is_empty());
    assert!(e.execute("transcript.cutRepeats", json!({})).is_err());
    // empty and punctuation-only words
    let w = spoken("— — … So so so so … —");
    assert!(find_repeats(&w, &RepeatOptions::default()).is_empty());
    assert!(find_repeats(&[], &RepeatOptions::default()).is_empty());
}
