//! The Timeline's sequence tabs: closing the others, reordering, and what a project file keeps of
//! them (the open tabs in order, the active one, and each sequence's zoom, scroll and track
//! heights). Premiere's behaviour was observed in Premiere Pro 26.5.2.

use super::*;
use filmcraft_project::SequenceView;
use serde_json::json;

/// The demo project with two more sequences open: tabs [main, b, c], `c` active.
fn three_tabs() -> (Session, [ItemId; 3]) {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let main = s.state.active_sequence.unwrap();
    let mut new = |name: &str| ItemId(s.execute("file.newSequence", json!({"name": name})).unwrap()["sequence"].as_u64().unwrap());
    let (b, c) = (new("B"), new("C"));
    assert_eq!(s.state.open_sequences, [main, b, c]);
    assert_eq!(s.state.active_sequence, Some(c));
    (s, [main, b, c])
}

fn view(pps: f64, scroll: f64) -> SequenceView {
    SequenceView { pps, scroll, v_scroll: 12.0, a_scroll: 7.0, video_track_h: 90.0, audio_track_h: 40.0, track_heights: Default::default() }
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("filmcraft-tabs-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn close_others_keeps_one_tab_and_shows_it() {
    let (mut s, [main, b, c]) = three_tabs();
    // from the active tab
    assert_eq!(s.execute("sequence.closeOthers", json!({})).unwrap()["closed"], json!(2));
    assert_eq!((s.state.open_sequences.clone(), s.state.active_sequence), (vec![c], Some(c)));
    // from another tab: that one becomes the active one
    let (mut s, _) = three_tabs();
    s.drain_events();
    s.execute("sequence.closeOthers", json!({"item": b.0})).unwrap();
    assert_eq!((s.state.open_sequences.clone(), s.state.active_sequence), (vec![b], Some(b)));
    assert!(s.drain_events().contains(&Event::OpenSequence(b)));
    // a sequence that is not open cannot be the one to keep
    s.execute("sequence.close", json!({})).unwrap();
    assert!(s.execute("sequence.closeOthers", json!({"item": main.0})).is_err());
}

#[test]
fn a_tab_moves_to_another_place_among_the_tabs() {
    let (mut s, [main, b, c]) = three_tabs();
    let (revision, steps) = (s.revision, s.history.undo.len());
    s.execute("sequence.moveTab", json!({"item": c.0, "index": 0})).unwrap();
    assert_eq!(s.state.open_sequences, [c, main, b]);
    assert_eq!(s.state.active_sequence, Some(c), "moving a tab does not change which is shown");
    s.execute("sequence.moveTab", json!({"item": c.0, "index": 1})).unwrap();
    assert_eq!(s.state.open_sequences, [main, c, b]);
    // past the end is the end; the active tab is the default
    assert_eq!(s.execute("sequence.moveTab", json!({"index": u64::MAX})).unwrap()["index"], json!(2));
    assert_eq!(s.state.open_sequences, [main, b, c]);
    // not an undo step, not an edit
    assert_eq!((s.revision, s.history.undo.len()), (revision, steps));
    s.execute("sequence.close", json!({"item": b.0})).unwrap();
    assert!(s.execute("sequence.moveTab", json!({"item": b.0, "index": 0})).is_err());
    assert!(s.execute("sequence.moveTab", json!({"item": c.0})).is_err());
}

#[test]
fn a_saved_project_reopens_with_its_tabs_and_their_views() {
    let (mut s, [main, b, c]) = three_tabs();
    s.execute("sequence.moveTab", json!({"item": c.0, "index": 0})).unwrap();
    s.execute("sequence.open", json!({"item": b.0})).unwrap();
    s.state.timeline_views.insert(main, view(12.5, 3.0));
    s.state.timeline_views.insert(b, view(400.0, 0.25));
    let dir = temp_dir("reopen");
    let path = dir.join("tabs.fcproj").to_string_lossy().to_string();
    s.execute("file.saveAs", json!({"path": path})).unwrap();

    let mut t = Session::default();
    t.execute("file.open", json!({"path": path})).unwrap();
    assert_eq!(t.state.open_sequences, [c, main, b]);
    assert_eq!(t.state.active_sequence, Some(b));
    assert_eq!(t.state.timeline_views.get(&main), Some(&view(12.5, 3.0)));
    assert_eq!(t.state.timeline_views.get(&b), Some(&view(400.0, 0.25)));
    assert_eq!(t.state.timeline_views.get(&c), None, "a sequence that was never shown has no view");
    assert!(t.drain_events().contains(&Event::OpenSequence(b)));
    assert!(!t.is_dirty());

    // with the preference off, a project opens on its first sequence as before
    let mut u = Session::default();
    u.prefs.timeline.restore_open_sequences = false;
    u.execute("file.open", json!({"path": path})).unwrap();
    assert_eq!(u.state.open_sequences, [main]);
    assert_eq!(u.state.active_sequence, Some(main));
    assert!(u.state.timeline_views.is_empty());

    // saved with every tab closed (or by a session that never showed a sequence): the project
    // opens on its first sequence, not on an empty Timeline, and still knows the views
    s.execute("sequence.closeOthers", json!({})).unwrap();
    s.execute("sequence.close", json!({})).unwrap();
    s.execute("file.save", json!({})).unwrap();
    let mut w = Session::default();
    w.execute("file.open", json!({"path": path})).unwrap();
    assert_eq!((w.state.open_sequences.clone(), w.state.active_sequence), (vec![main], Some(main)));
    assert_eq!(w.state.timeline_views.get(&b), Some(&view(400.0, 0.25)));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The view in a project file is not trusted: ids of things that are not sequences, repeats and
/// numbers out of range are dropped or brought into range, and a view that cannot be read at all
/// does not stop the project from opening.
#[test]
fn a_damaged_view_in_a_project_file_is_cleaned_up_or_ignored() {
    let (mut s, [main, b, c]) = three_tabs();
    let footage = s.project.sequence(main).unwrap().video_tracks[0].items[0].item;
    let dir = temp_dir("damaged");
    let path = dir.join("tabs.fcproj").to_string_lossy().to_string();
    s.execute("file.saveAs", json!({"path": path})).unwrap();
    let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let with_view = |view: serde_json::Value| {
        let mut doc = saved.clone();
        doc["view"] = view;
        std::fs::write(&path, serde_json::to_vec(&doc).unwrap()).unwrap();
        let mut t = Session::default();
        t.execute("file.open", json!({"path": path})).unwrap();
        t
    };

    let t = with_view(json!({
        "open_sequences": [c.0, 999_999, footage.0, c.0, b.0],
        "active_sequence": 999_999,
        "sequences": {
            b.0.to_string(): {"pps": 1e300, "scroll": -5.0, "v_scroll": 1e30, "a_scroll": -1.0, "video_track_h": 0.0, "audio_track_h": 1e9, "track_heights": {"7": 1e9, "8": -3.0, "9": 120.0}},
            "999999": {"pps": 40.0, "scroll": 0.0, "video_track_h": 60.0, "audio_track_h": 56.0},
            footage.0.to_string(): {"pps": 40.0, "scroll": 0.0, "video_track_h": 60.0, "audio_track_h": 56.0},
        },
    }));
    assert_eq!(t.state.open_sequences, [c, b]);
    assert_eq!(t.state.active_sequence, Some(c), "an active tab that is not open: the first tab");
    assert_eq!(t.state.timeline_views.len(), 1);
    let v = t.state.timeline_views[&b].clone();
    assert!(v.pps <= 1e5 && v.scroll == 0.0 && v.v_scroll <= 1e6 && v.a_scroll == 0.0);
    assert!(v.video_track_h >= 8.0 && v.audio_track_h <= 600.0);
    // tracks resized on their own: kept, brought into range
    assert_eq!(v.track_heights, [(7, 600.0), (8, 8.0), (9, 120.0)].into());

    // not a view at all: the project opens as it does without one
    for junk in [json!("tabs"), json!([1, 2, 3]), json!({"open_sequences": "all"}), json!({"sequences": {"x": 1}}), json!(null)] {
        let t = with_view(junk.clone());
        assert_eq!((t.state.open_sequences.clone(), t.state.active_sequence), (vec![main], Some(main)), "{junk}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_view_of_a_deleted_sequence_is_forgotten() {
    let (mut s, [main, b, c]) = three_tabs();
    for id in [main, b, c] {
        s.state.timeline_views.insert(id, view(40.0, 0.0));
    }
    s.execute("project.delete", json!({"items": [b.0]})).unwrap();
    assert_eq!(s.state.timeline_views.keys().copied().collect::<Vec<_>>(), [main, c]);
    assert_eq!(s.state.open_sequences, [main, c]);
    // the saved view never names it either
    assert_eq!(s.project_view().open_sequences, [main, c]);
}
