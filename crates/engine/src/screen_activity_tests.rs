use serde_json::json;

use crate::media_test_util::{make_movie, session_with, tmp_dir};
use filmcraft_media::DemoScene;

#[test]
fn moving_footage_is_active_and_a_frame_hold_is_still() {
    let dir = tmp_dir("screen-activity");
    let path = dir.join("plasma.mov");
    make_movie(&path, DemoScene::Plasma, 160, 90, 48);
    let (mut s, _, _) = session_with(&[&path]);
    let v = s.active_sequence().unwrap().video_tracks[0].items[0].id;

    // the plasma drifts smoothly (about 1.5 grey levels a step): count any change
    let r = s.execute("video.screenActivity", json!({"threshold": 1})).unwrap();
    assert_eq!(r["clips"], json!([v.0]), "the V1 clip: {r}");
    let samples = r["samples"].as_array().unwrap();
    assert!(samples.len() >= 18, "2 s every 100 ms: {}", samples.len());
    assert!(r["activeSeconds"].as_f64().unwrap() > 1.5, "{r}");
    assert_eq!(r["activeRanges"].as_array().unwrap().len(), 1, "one continuous range: {r}");
    assert_eq!(r["strip"].as_str().unwrap().chars().count(), samples.len());

    // a region and a sub-range
    let r = s
        .execute(
            "video.screenActivity",
            json!({"clip": v.0, "from": 0.5, "to": 1.0, "region": {"x": 0, "y": 0, "w": 80, "h": 45}, "threshold": 1, "samples": false}),
        )
        .unwrap();
    assert!(r["samples"].is_null() && r["activeSeconds"].as_f64().unwrap() > 0.3, "{r}");

    // frozen: nothing moves
    s.edit_sequence("hold", |q, _, _| {
        let it = &mut q.video_tracks[0].items[0];
        it.frame_hold = Some(it.source_in);
        Ok(())
    })
    .unwrap();
    let r = s.execute("video.screenActivity", json!({"threshold": 1})).unwrap();
    assert_eq!(r["activeSeconds"], 0.0, "{r}");
    assert!(r["strip"].as_str().unwrap().chars().all(|c| c == '.'), "{r}");

    // across a cut: both pieces are measured, the step on the cut is not compared
    s.edit_sequence("unhold", |q, _, _| {
        q.video_tracks[0].items[0].frame_hold = None;
        Ok(())
    })
    .unwrap();
    s.edit_sequence("cut", |q, ctx, _| {
        let tracks: Vec<_> = q.all_tracks().map(|t| t.id).collect();
        let r = filmcraft_time::TimeRange::from_bounds(filmcraft_time::Tick::from_seconds_f64(0.95), filmcraft_time::Tick::from_seconds_f64(1.05));
        filmcraft_edit::extract(q, &tracks, r, ctx);
        Ok(())
    })
    .unwrap();
    let r = s.execute("video.screenActivity", json!({"threshold": 1})).unwrap();
    assert_eq!(r["clips"].as_array().unwrap().len(), 2, "{r}");
    assert_eq!(r["strip"].as_str().unwrap().matches('|').count(), 1, "{r}");
    assert!(r["activeSeconds"].as_f64().unwrap() > 1.4, "{r}");

    assert!(s.execute("video.screenActivity", json!({"clip": 999_999})).is_err());
    assert!(s.execute("video.screenActivity", json!({"stepMs": 10, "from": 0, "to": 2.0})).is_ok(), "200 steps is fine");
}
