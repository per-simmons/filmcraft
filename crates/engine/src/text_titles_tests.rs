use serde_json::{Value, json};

use crate::Session;
use filmcraft_time::{FrameRate, Tick};

/// An empty 1920×1080 60 fps sequence.
fn seq() -> Session {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"name": "Titles", "fps": 60, "width": 1920, "height": 1080, "video": 1, "audio": 1})).unwrap();
    s
}

fn frame(f: i64) -> Tick {
    FrameRate::FPS_60.tick_of(f)
}

/// The sequence frame at `t`, full size.
fn render(s: &Session, t: Tick) -> filmcraft_render::Image {
    let provider = s.media.full_res_provider(s.project.clone(), s.services.clone());
    let opts = filmcraft_render::RenderOptions { scale: 1.0, ..Default::default() };
    filmcraft_render::render_sequence(&s.project, s.state.active_sequence.unwrap(), t, opts, &provider)
}

fn rect(v: &Value) -> [f64; 4] {
    let a = v.as_array().unwrap();
    [a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap(), a[3].as_f64().unwrap()]
}

/// Brightest luma inside a sequence-pixel rectangle (shrunk by `inset` px).
fn max_luma(img: &filmcraft_render::Image, r: [f64; 4], inset: f64) -> f32 {
    let mut m = 0.0f32;
    let (x0, y0) = ((r[0] + inset).max(0.0) as usize, (r[1] + inset).max(0.0) as usize);
    let (x1, y1) = (((r[0] + r[2] - inset) as usize).min(img.w), ((r[1] + r[3] - inset) as usize).min(img.h));
    for y in y0..y1 {
        for x in x0..x1 {
            let p = img.get(x, y);
            m = m.max(p[0].max(p[1]).max(p[2]));
        }
    }
    m
}

#[test]
fn emphasis_title_is_one_white_caps_line_low_and_centred() {
    let mut s = seq();
    let undo = s.history.undo.len();
    let r = s.execute("graphics.emphasisTitle", json!({"text": "no experience required", "font": "Inter", "fontStyle": "Bold"})).unwrap();
    assert_eq!(s.history.undo.len(), undo + 1, "one undo step");
    assert_eq!(r["text"], "NO EXPERIENCE REQUIRED");
    assert_eq!(r["lines"].as_array().unwrap().len(), 1, "a short phrase stays on one line: {r}");
    assert_eq!(r["size"], 111.0, "222 at 2160 scales to 111 at 1080");
    let l = &r["layout"];
    let words: Vec<&str> = l["units"].as_array().unwrap().iter().map(|u| u["text"].as_str().unwrap()).collect();
    assert_eq!(words, vec!["NO", "EXPERIENCE", "REQUIRED"]);
    let block = rect(&l["block"]);
    let cx = block[0] + block[2] / 2.0;
    assert!((cx - 960.0).abs() < 2.0, "centred: {block:?}");
    assert!(block[1] > 1080.0 * 0.6 && block[1] + block[3] < 1080.0 * 0.9, "below the face, above the player controls: {block:?}");
    assert_eq!(l["safe"]["titleSafe"], true);
    // white, no box: the middle of the frame is untouched and a held word is white
    let clip = r["clip"].as_u64().unwrap();
    let hold = render(&s, frame(120));
    let w0 = rect(&l["units"][1]["rect"]);
    assert!(max_luma(&hold, w0, 0.0) > 0.95, "white text");
    assert_eq!(hold.get(960, 300)[3], 0.0, "no box over the picture");
    // one undo removes the whole title
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.active_sequence().unwrap().find_item(filmcraft_project::ClipId(clip)).is_none());
}

#[test]
fn long_titles_get_balanced_lines_without_an_orphan() {
    let mut s = seq();
    let r = s
        .execute(
            "graphics.emphasisTitle",
            json!({"text": "you do not need any coding experience to build this", "placement": "upperLeft", "font": "Inter", "fontStyle": "Bold", "size": 64}),
        )
        .unwrap();
    let lines = r["lines"].as_array().unwrap();
    assert!(lines.len() >= 2, "{r}");
    let bal = &r["layout"]["balance"];
    assert!(bal["ratio"].as_f64().unwrap() > 0.6, "balanced: {bal}");
    assert_eq!(bal["orphan"], false);
    assert_eq!(r["fits"], true, "{r}");
    // upper left: left-aligned near the left edge, in the upper part of the frame
    let block = rect(&r["layout"]["block"]);
    assert!(block[0] > 100.0 && block[0] < 160.0 && block[1] > 150.0 && block[1] < 220.0, "{block:?}");
    assert!(block[0] + block[2] < 1920.0 * 0.5, "{block:?}");
    // upper right mirrors it
    let r = s
        .execute("graphics.emphasisTitle", json!({"text": "upper right title here", "placement": "upperRight", "font": "Inter", "size": 90, "newClip": true}))
        .unwrap();
    let b = rect(&r["layout"]["block"]);
    assert!((b[0] + b[2] - 1920.0 * 0.93).abs() < 3.0, "{b:?}");
}

#[test]
fn words_render_one_by_one_and_the_layout_reports_them() {
    let mut s = seq();
    let r = s.execute("graphics.emphasisTitle", json!({"text": "no experience required", "font": "Inter", "fontStyle": "Bold"})).unwrap();
    let clip = r["clip"].as_u64().unwrap();
    let rest = r["layout"]["units"].as_array().unwrap().iter().map(|u| rect(&u["rect"])).collect::<Vec<_>>();
    // the schedule: 14-frame offsets, 20 frames each, landing at frame 48
    let units = r["animation"]["units"].as_array().unwrap();
    let starts: Vec<i64> = units.iter().map(|u| u["intro"]["seqStartFrame"].as_i64().unwrap()).collect();
    assert_eq!(starts, vec![0, 14, 28]);
    assert_eq!(r["animation"]["introEndSeq"].as_i64().unwrap(), frame(48).0);
    // frame 8: NO is rising in, the others have not started
    let at = |s: &mut Session, f: i64| s.execute("graphics.textLayout", json!({"clip": clip, "animated": true, "frame": f})).unwrap();
    let l8 = at(&mut s, 8);
    let ops: Vec<f64> = l8["units"].as_array().unwrap().iter().map(|u| u["opacity"].as_f64().unwrap()).collect();
    assert!(ops[0] > 0.05 && ops[0] < 1.0 && ops[1] == 0.0 && ops[2] == 0.0, "{ops:?}");
    let no8 = rect(&l8["units"][0]["rect"]);
    assert!(no8[1] > rest[0][1] + 3.0, "still below its place: {no8:?} vs {:?}", rest[0]);
    let img = render(&s, frame(8));
    assert!(max_luma(&img, rest[0], 0.0) > 0.05);
    assert_eq!(max_luma(&img, rest[1], 2.0), 0.0, "EXPERIENCE not drawn yet");
    assert_eq!(max_luma(&img, rest[2], 2.0), 0.0, "REQUIRED not drawn yet");
    // frame 34: NO and EXPERIENCE settled, REQUIRED arriving; the boxes never re-centre
    let img = render(&s, frame(34));
    assert!(max_luma(&img, rest[1], 0.0) > 0.95);
    let l34 = at(&mut s, 34);
    assert_eq!(rect(&l34["units"][1]["rect"]), rest[1]);
    assert!(l34["units"][2]["opacity"].as_f64().unwrap() < 1.0);
    // at rest (not animated) the boxes are the same at any time
    let still = s.execute("graphics.textLayout", json!({"clip": clip, "frame": 8})).unwrap();
    assert_eq!(rect(&still["units"][2]["rect"]), rest[2]);
    // characters and lines
    let chars = s.execute("graphics.textLayout", json!({"clip": clip, "unit": "char", "frame": 120})).unwrap();
    assert_eq!(chars["units"].as_array().unwrap().len(), 20);
    let lines = s.execute("graphics.textLayout", json!({"clip": clip, "unit": "line", "frame": 120})).unwrap();
    assert_eq!(lines["units"].as_array().unwrap().len(), 1);
}

#[test]
fn animate_text_takes_frames_and_units() {
    let mut s = seq();
    let r = s.execute("graphics.newText", json!({"text": "ONE TWO\nTHREE", "font": "Inter", "size": 80})).unwrap();
    let clip = r["clip"].as_u64().unwrap();
    let a = s
        .execute(
            "graphics.animateText",
            json!({"clip": clip, "unit": "line", "staggerFrames": 10, "durationFrames": 24, "delayFrames": 6, "rise": 50, "scale": 80, "easeIn": 0, "easeOut": 100, "outro": {"durationFrames": 12, "staggerFrames": 6, "rise": 20}}),
        )
        .unwrap();
    assert_eq!(a["unitCount"], 2);
    let u = a["units"].as_array().unwrap();
    assert_eq!(u[0]["intro"]["seqStartFrame"], 6);
    assert_eq!(u[1]["intro"]["seqStartFrame"], 16);
    assert_eq!(u[1]["intro"]["seqEndFrame"], 40);
    // the clip is 5 s = 300 frames: exits end at the clip end, 6 frames apart
    assert_eq!(u[1]["outro"]["seqEndFrame"], 300);
    assert_eq!(u[0]["outro"]["seqStartFrame"], 300 - 12 - 6);
    let anim = &a["animation"];
    assert_eq!(anim["unit"], "line");
    assert_eq!(anim["intro"]["offset"][1], 50.0);
    assert_eq!(anim["outro"]["offset"][1], -20.0, "an exit rise moves up");
    // totalFrames spreads the starts so the last unit lands then
    let a = s.execute("graphics.animateText", json!({"clip": clip, "preset": "word-rise", "totalFrames": 60, "durationFrames": 20})).unwrap();
    let ends: Vec<i64> = a["units"].as_array().unwrap().iter().map(|u| u["intro"]["seqEndFrame"].as_i64().unwrap()).collect();
    assert_eq!(*ends.last().unwrap(), 60, "{ends:?}");
    // turning it off removes the animation; undo brings it back
    let off = s.execute("graphics.animateText", json!({"clip": clip, "off": true})).unwrap();
    assert!(off["animation"].is_null());
    s.execute("edit.undo", json!({})).unwrap();
    let again = s.execute("graphics.textLayout", json!({"clip": clip, "animated": true, "frame": 0})).unwrap();
    assert_eq!(again["units"][0]["opacity"], 0.0);
    // the presets are listed
    let p = s.execute("graphics.textAnimationPresets", json!({})).unwrap();
    assert!(p.as_array().unwrap().iter().any(|x| x["id"] == "word-rise"));
}

#[test]
fn animation_survives_save_and_copy() {
    let mut s = seq();
    let r = s.execute("graphics.emphasisTitle", json!({"text": "saved", "font": "Inter"})).unwrap();
    let clip = filmcraft_project::ClipId(r["clip"].as_u64().unwrap());
    let json = serde_json::to_string(&s.project).unwrap();
    let back: filmcraft_project::Project = serde_json::from_str(&json).unwrap();
    let q = back.sequence(s.state.active_sequence.unwrap()).unwrap();
    let (_, it) = q.find_item(clip).unwrap();
    let anim = it.effects.iter().find_map(|e| e.layer.as_ref().and_then(|x| x.anim.clone())).expect("stored with the project");
    assert_eq!(anim, filmcraft_project::text_anim::preset("word-rise").unwrap());
}

#[test]
fn hostile_parameters_are_errors_not_crashes() {
    let mut s = seq();
    let r = s.execute("graphics.newText", json!({"text": "HOSTILE", "font": "Inter"})).unwrap();
    let clip = r["clip"].as_u64().unwrap();
    for bad in [json!({"clip": clip, "unit": "paragraphs"}), json!({"clip": clip, "order": "sideways"}), json!({"clip": clip, "preset": "nope"})] {
        assert!(s.execute("graphics.animateText", bad.clone()).is_err(), "{bad}");
    }
    for odd in [
        json!({"clip": clip, "staggerFrames": 1e300, "durationFrames": -5, "rise": 1e308, "scale": -1, "easeIn": 1e9}),
        json!({"clip": clip, "stagger": i64::MAX, "duration": i64::MIN, "delay": i64::MAX}),
        json!({"clip": clip, "totalFrames": 0, "unit": "char"}),
        json!({"clip": clip, "fade": "lots", "seed": u64::MAX}),
    ] {
        s.execute("graphics.animateText", odd.clone()).unwrap_or_else(|e| panic!("{odd}: {e}"));
        for f in [0, 1, 299, 300] {
            s.execute("graphics.textLayout", json!({"clip": clip, "animated": true, "frame": f})).unwrap();
            let _ = render(&s, frame(f.min(299)));
        }
    }
    assert!(s.execute("graphics.textLayout", json!({"clip": clip, "unit": "x"})).is_err());
    assert!(s.execute("graphics.emphasisTitle", json!({"text": "   "})).is_err());
    assert!(s.execute("graphics.emphasisTitle", json!({"text": "x", "placement": "behind"})).is_err());
    assert!(s.execute("graphics.emphasisTitle", json!({"text": "x", "size": f64::MAX, "maxLines": 0, "maxWidth": -4})).is_ok());
    // a shape layer is not text
    let r = s.execute("graphics.newShape", json!({"shape": "rectangle", "clip": clip})).unwrap();
    assert!(s.execute("graphics.textLayout", json!({"clip": clip, "layer": r["layer"]})).is_err());
}

#[test]
fn emphasis_templates_carry_the_look_and_the_word_rise() {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"name": "4K", "fps": 60, "width": 3840, "height": 2160, "video": 1, "audio": 1})).unwrap();
    for (name, rises) in [("Emphasis – Serif Word Rise", true), ("Emphasis – Serif Caps", false)] {
        let r = s.execute("graphics.template.apply", json!({"template": name, "time": 0})).unwrap();
        let clip = r["clip"].as_u64().unwrap();
        let l = s.execute("graphics.textLayout", json!({"clip": clip, "animated": true, "frame": 8})).unwrap();
        let words: Vec<&str> = l["units"].as_array().unwrap().iter().map(|u| u["text"].as_str().unwrap()).collect();
        assert_eq!(words, vec!["NO", "EXPERIENCE", "REQUIRED"]);
        let ops: Vec<f64> = l["units"].as_array().unwrap().iter().map(|u| u["opacity"].as_f64().unwrap()).collect();
        if rises {
            assert!(ops[0] > 0.0 && ops[0] < 1.0 && ops[2] == 0.0, "{ops:?}");
        } else {
            assert_eq!(ops, vec![1.0, 1.0, 1.0]);
        }
        // one centred line low in the 4K frame (designed at 1080p, scaled ×2)
        let b = rect(&l["block"]);
        assert!((b[0] + b[2] / 2.0 - 1920.0).abs() < 4.0 && b[1] > 1400.0 && b[1] + b[3] < 1950.0, "{b:?}");
        assert_eq!(l["lines"].as_array().unwrap().len(), 1);
        s.execute("edit.undo", json!({})).unwrap();
    }
}

#[test]
fn default_emphasis_font_is_a_serif() {
    let mut s = seq();
    let r = s.execute("graphics.emphasisTitle", json!({"text": "serif check"})).unwrap();
    let font = r["font"].as_str().unwrap();
    assert!(font == "Bodoni 72" || font == "Noto Serif", "{font}");
    assert_eq!(r["fontFallback"], font == "Noto Serif");
    assert_eq!(r["layout"]["fontMissing"], false, "the chosen font resolves");
}
