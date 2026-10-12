//! Keyboard shortcuts: chords, presets, conflicts, panel scope, undo, files.

use super::*;
use crate::shortcuts::{Binding, Chord, CommandInfo, Platform, Shortcuts};
use serde_json::json;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("fc-shortcuts-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The frontend's commands, as the egui UI registers them (a few are enough here).
fn with_ui_commands(s: &mut Session) {
    s.shortcuts.register_external(vec![
        CommandInfo::new("playback.toggle", "Play/Stop", &[], Some("Space")),
        CommandInfo::new("playback.forward", "Shuttle Right", &[], Some("L")),
        CommandInfo::new("playback.slowForward", "Shuttle Slow Right", &[], None),
        CommandInfo::new("playback.playAround", "Play Around", &[], None),
        CommandInfo::new("tool.razor", "Razor Tool", &[], Some("C")),
        CommandInfo::new("app.keyboardShortcuts", "Keyboard Shortcuts…", &["Edit"], Some("Cmd+Alt+K")),
    ]);
}

#[test]
fn chords_parse_normalise_and_display_per_os() {
    let c = Chord::parse("⌥⇧⌘K").unwrap();
    assert_eq!(c.canonical(), "Cmd+Alt+Shift+K");
    assert_eq!(Chord::parse("shift+opt+command+k").unwrap(), c);
    assert_eq!(c.display(Platform::Mac), "⌥⇧⌘K");
    assert_eq!(c.display(Platform::Windows), "Ctrl+Alt+Shift+K");
    assert_eq!(Chord::parse("Return").unwrap().canonical(), "Enter");
    assert_eq!(Chord::parse("Forward Delete").unwrap().canonical(), "Delete");
    assert_eq!(Chord::parse("Shift+Left").unwrap().display(Platform::Mac), "⇧←");
    // Windows text: Ctrl is the primary modifier
    assert_eq!(Chord::parse_for("Ctrl+Shift+Z", Platform::Windows).unwrap().canonical(), "Cmd+Shift+Z");
    assert_eq!(Chord::parse_for("Win+T", Platform::Windows).unwrap().canonical(), "Ctrl+T");
    // macOS Control folds into Ctrl elsewhere
    assert_eq!(Chord::parse("Ctrl+T").unwrap().effective(Platform::Windows), Chord::parse("Cmd+T").unwrap());
    assert!(Chord::parse("Cmd+").is_err());
    assert!(Chord::parse("Hyper+K").is_err());
}

#[test]
fn defaults_follow_the_registry_and_the_premiere_audit() {
    let mut s = Session::default();
    with_ui_commands(&mut s);
    let sc = &s.shortcuts;
    assert_eq!(sc.preset, "FilmCraft Default");
    assert_eq!(sc.primary("sequence.addEdit").as_deref(), Some("Cmd+K"));
    assert_eq!(sc.primary("playback.toggle").as_deref(), Some("Space"));
    // the audit adopts Premiere defaults for commands that had none
    let audit = sc.audit();
    let has = |c: &str, k: &str, p: Option<&str>| audit.iter().any(|b| b.command == c && b.keys == k && b.panel.as_deref() == p);
    assert!(has("trim.applyDefaultTransition", "Shift+D", None), "{audit:?}");
    assert!(has("playback.slowForward", "Shift+L", None));
    assert!(has("playback.playAround", "Shift+K", None));
    assert!(has("source.open", "Shift+O", Some("Project")));
    assert_eq!(sc.primary("trim.applyDefaultTransition").as_deref(), Some("Shift+D"));
    // no conflicts in any built-in preset (macOS)
    for name in crate::shortcuts::BUILTIN_PRESETS {
        let mut t = Session::default();
        with_ui_commands(&mut t);
        t.shortcuts.load_preset(name).unwrap();
        assert!(t.shortcuts.conflicts(Platform::Mac).is_empty(), "{name}: {:?}", t.shortcuts.conflicts(Platform::Mac));
    }
    // command.list reports the live bindings
    let list = s.execute("command.list", json!({})).unwrap();
    let ae = list.as_array().unwrap().iter().find(|c| c["id"] == "sequence.addEdit").unwrap();
    assert_eq!(ae["shortcut"], json!("Cmd+K"));
}

#[test]
fn assign_reassigns_conflicts_and_undoes() {
    let mut s = Session::default();
    let r = s.execute("shortcuts.set", json!({"command": "edit.undo", "keys": "Cmd+Shift+K"})).unwrap();
    assert_eq!(r["reassigned"][0]["command"], json!("sequence.addEditAllTracks"), "{r}");
    assert!(s.shortcuts.primary("sequence.addEditAllTracks").is_none());
    assert_eq!(s.shortcuts.primary("edit.undo").as_deref(), Some("Cmd+Shift+K"), "replaces Cmd+Z");
    let hit = s.execute("shortcuts.resolve", json!({"keys": "⇧⌘K"})).unwrap();
    assert_eq!(hit["command"], json!("edit.undo"));
    assert!(s.shortcuts.modified);
    // add a second shortcut, keep a conflict on purpose
    s.execute("shortcuts.set", json!({"command": "edit.undo", "keys": "Cmd+Z", "add": true})).unwrap();
    assert_eq!(s.shortcuts.for_command("edit.undo").iter().filter(|b| b.panel.is_none()).count(), 2);
    let r = s.execute("shortcuts.set", json!({"command": "edit.redo", "keys": "M", "keepConflicts": true})).unwrap();
    assert_eq!(r["conflicts"].as_array().unwrap().len(), 1, "{r}");
    let c = s.execute("shortcuts.conflicts", json!({"platform": "mac"})).unwrap();
    assert_eq!(c["conflicts"][0]["keys"], json!("M"));
    // undo / redo within the editor
    s.execute("shortcuts.undo", json!({})).unwrap();
    assert!(s.execute("shortcuts.conflicts", json!({})).unwrap()["conflicts"].as_array().unwrap().is_empty());
    s.execute("shortcuts.undo", json!({})).unwrap();
    s.execute("shortcuts.undo", json!({})).unwrap();
    assert_eq!(s.shortcuts.primary("edit.undo").as_deref(), Some("Cmd+Z"));
    assert_eq!(s.shortcuts.primary("sequence.addEditAllTracks").as_deref(), Some("Cmd+Shift+K"));
    s.execute("shortcuts.redo", json!({})).unwrap();
    assert_eq!(s.shortcuts.primary("edit.undo").as_deref(), Some("Cmd+Shift+K"));
    // clear
    let r = s.execute("shortcuts.clear", json!({"command": "edit.undo"})).unwrap();
    assert_eq!(r["removed"], json!(2), "Cmd+Shift+K and History ▸ Left");
    assert!(s.shortcuts.for_command("edit.undo").is_empty());
    assert!(s.execute("shortcuts.set", json!({"command": "no.such", "keys": "K"})).is_err());
    assert!(s.execute("shortcuts.set", json!({"command": "edit.undo", "keys": "K", "panel": "Nowhere"})).is_err());
    // editor Cancel restores the set it opened with
    s.shortcuts.begin_editing();
    s.execute("shortcuts.set", json!({"command": "edit.undo", "keys": "U"})).unwrap();
    s.shortcuts.cancel_editing();
    assert!(s.shortcuts.for_command("edit.undo").is_empty());
}

#[test]
fn panel_shortcuts_override_application_ones_with_focus() {
    let s = Session::default();
    let left = Chord::parse("Left").unwrap();
    let p = Platform::Mac;
    assert_eq!(s.shortcuts.resolve(&left, None, p).unwrap().command, "playhead.stepBack");
    assert_eq!(s.shortcuts.resolve(&left, Some("Timeline"), p).unwrap().command, "playhead.stepBack");
    assert_eq!(s.shortcuts.resolve(&left, Some("History"), p).unwrap().command, "edit.undo");
    let bs = Chord::parse("Backspace").unwrap();
    assert_eq!(s.shortcuts.resolve(&bs, Some("Project"), p).unwrap().command, "project.delete");
    assert_eq!(s.shortcuts.resolve(&bs, None, p).unwrap().command, "edit.clear");
    let ov = s.shortcuts.overrides(p);
    assert!(ov.iter().any(|(panel, k, cmd, app)| panel == "History" && k == "Left" && cmd == "edit.undo" && app == "playhead.stepBack"), "{ov:?}");
    // Windows: macOS-Control shortcuts fold into Ctrl and can clash
    let mut t = Session::default();
    t.execute("shortcuts.set", json!({"command": "edit.copy", "keys": "Ctrl+C", "add": true, "keepConflicts": true})).unwrap();
    assert!(t.shortcuts.conflicts(Platform::Mac).is_empty());
    assert!(!t.shortcuts.conflicts(Platform::Windows).is_empty() || t.shortcuts.for_command("edit.copy").len() == 2);
    let k = t.execute("shortcuts.forKey", json!({"key": "C"})).unwrap();
    assert!(k.as_array().unwrap().iter().any(|b| b["command"] == "edit.copy" && b["keys"] == "Ctrl+C"), "{k}");
}

#[test]
fn delete_clears_timeline_clips_and_project_items() {
    // #243: with the Timeline focused, Delete (the forward-delete key, labelled Delete on Windows
    // and Linux keyboards) resolved to Project ▸ Clear and left the selected clips in place.
    let s = Session::default();
    let del = Chord::parse("Delete").unwrap();
    for p in [Platform::Mac, Platform::Windows] {
        assert_eq!(s.shortcuts.resolve(&del, Some("Timeline"), p).unwrap().command, "edit.clear", "{p:?}");
        assert_eq!(s.shortcuts.resolve(&del, Some("Project"), p).unwrap().command, "project.delete", "{p:?}");
        assert_eq!(s.shortcuts.resolve(&del, None, p).unwrap().command, "project.delete", "{p:?}");
    }
}

#[test]
fn compat_presets_map_other_editors() {
    let mut s = Session::default();
    with_ui_commands(&mut s);
    s.execute("shortcuts.loadPreset", json!({"name": "Premiere Pro Compatible"})).unwrap();
    assert_eq!(s.shortcuts.primary("markers.clearIn").as_deref(), Some("Alt+I"));
    assert_eq!(s.shortcuts.primary("edit.clear").as_deref(), Some("Delete"));
    assert!(s.shortcuts.for_command("edit.clear").iter().any(|b| b.panel.as_deref() == Some("Timeline") && b.keys == "Backspace"));
    s.execute("shortcuts.loadPreset", json!({"name": "Final Cut Pro Compatible"})).unwrap();
    assert_eq!(s.shortcuts.primary("source.insert").as_deref(), Some("W"));
    assert_eq!(s.shortcuts.primary("tool.razor").as_deref(), Some("B"));
    assert!(s.shortcuts.primary("trim.rippleNext").is_none(), "W was taken by Insert");
    s.execute("shortcuts.loadPreset", json!({"name": "Avid Media Composer Compatible"})).unwrap();
    assert_eq!(s.shortcuts.primary("sequence.lift").as_deref(), Some("Z"));
    assert_eq!(s.shortcuts.for_command("markers.markIn").len(), 2, "I and E");
    assert!(!s.shortcuts.modified);
    assert!(s.execute("shortcuts.loadPreset", json!({"name": "Nope"})).is_err());
    s.execute("shortcuts.loadPreset", json!({"name": "FilmCraft Default"})).unwrap();
    assert_eq!(s.shortcuts.primary("source.insert").as_deref(), Some(","));
}

#[test]
fn custom_presets_persist_export_and_import() {
    let dir = temp_dir("presets");
    let mut s = Session::default();
    s.shortcuts.set_dir(&dir);
    s.execute("shortcuts.set", json!({"command": "sequence.addEdit", "keys": "Cmd+Shift+B"})).unwrap();
    let r = s.execute("shortcuts.savePreset", json!({"name": "My Keys"})).unwrap();
    assert!(std::path::Path::new(r["path"].as_str().unwrap()).exists());
    let p = s.execute("shortcuts.presets", json!({})).unwrap();
    assert_eq!(p["active"], json!("My Keys"));
    assert_eq!(p["custom"], json!(["My Keys"]));
    assert!(s.execute("shortcuts.savePreset", json!({"name": "FilmCraft Default"})).is_err());
    // the active set survives a restart
    let mut t = Shortcuts::new();
    t.set_dir(&dir);
    assert_eq!(t.preset, "My Keys");
    assert_eq!(t.primary("sequence.addEdit").as_deref(), Some("Cmd+Shift+B"));
    // switch away and back
    s.execute("shortcuts.loadPreset", json!({"name": "FilmCraft Default"})).unwrap();
    assert_eq!(s.shortcuts.primary("sequence.addEdit").as_deref(), Some("Cmd+K"));
    s.execute("shortcuts.loadPreset", json!({"name": "My Keys"})).unwrap();
    assert_eq!(s.shortcuts.primary("sequence.addEdit").as_deref(), Some("Cmd+Shift+B"));
    // export, then import a Windows-notation file
    let out = dir.join("export.json");
    s.execute("shortcuts.export", json!({"path": out.to_string_lossy()})).unwrap();
    let f: crate::shortcuts::PresetFile = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert!(f.bindings.contains(&Binding { command: "sequence.addEdit".into(), keys: "Cmd+Shift+B".into(), panel: None }));
    let win = dir.join("win.json");
    std::fs::write(
        &win,
        serde_json::to_vec(&json!({"format": "filmcraft-keyboard-shortcuts", "version": 1, "name": "From Windows", "platform": "windows",
            "bindings": [{"command": "edit.undo", "keys": "Ctrl+Z"}, {"command": "sequence.addEdit", "keys": "Ctrl+Alt+K"}, {"command": "no.such", "keys": "K"}]}))
        .unwrap(),
    )
    .unwrap();
    let r = s.execute("shortcuts.import", json!({"path": win.to_string_lossy()})).unwrap();
    assert_eq!(r["name"], json!("From Windows"));
    assert_eq!(s.shortcuts.primary("edit.undo").as_deref(), Some("Cmd+Z"));
    assert_eq!(s.shortcuts.primary("sequence.addEdit").as_deref(), Some("Cmd+Alt+K"));
    assert_eq!(s.shortcuts.bindings.len(), 2, "unknown commands dropped");
    s.execute("shortcuts.deletePreset", json!({"name": "My Keys"})).unwrap();
    assert!(s.execute("shortcuts.deletePreset", json!({"name": "FilmCraft Default"})).is_err());
    assert_eq!(s.execute("shortcuts.presets", json!({})).unwrap()["custom"], json!(["From Windows"]));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn list_searches_labels_and_keys() {
    let s = Session::default();
    let r = s.shortcuts.commands();
    assert!(r.iter().any(|c| c.id == "trim.forward"));
    assert!(!r.iter().any(|c| c.id == "timeline.move"), "commands with required params are not bindable");
    // a parameter doc holds parameters only (each optional or with a default): a result
    // description in it would take the command out of this list
    for id in ["clip.replaceFromBin", "clip.replaceFromSource", "clip.replaceFromSourceMatchFrame"] {
        assert!(r.iter().any(|c| c.id == id), "{id} has no required parameter, so it can be bound");
    }
    let mut s = s;
    let v = s.execute("shortcuts.list", json!({"query": "ripple delete"})).unwrap();
    assert!(v.as_array().unwrap().iter().any(|c| c["id"] == "edit.rippleDelete"), "{v}");
    let v = s.execute("shortcuts.list", json!({"query": "cmd+k", "platform": "windows"})).unwrap();
    assert!(v.as_array().unwrap().iter().any(|c| c["id"] == "sequence.addEdit"), "{v}");
    let v = s.execute("shortcuts.list", json!({"panel": "History", "assigned": true})).unwrap();
    assert_eq!(v.as_array().unwrap().len(), 2, "{v}");
}

// ------------------------------------------------------------------ Premiere .kys files

/// One `.kys` item: `key` is a Premiere virtual key code (0 = no key), `mods` = "cmd opt shift ctrl".
fn kys_item(i: usize, cmd: &str, key: u64, mods: &str) -> String {
    let m = |n: &str| mods.split_whitespace().any(|x| x == n);
    if key == 0 {
        return format!("<item.{i} Version=\"1\"><commandname>{cmd}</commandname></item.{i}>");
    }
    format!(
        "<item.{i} Version=\"1\"><virtualkey>{key}</virtualkey><modifier.command>{}</modifier.command><modifier.opt>{}</modifier.opt>\
         <modifier.shift>{}</modifier.shift><modifier.ctrl>{}</modifier.ctrl><commandname>{cmd}</commandname></item.{i}>",
        m("cmd"),
        m("opt"),
        m("shift"),
        m("ctrl")
    )
}

fn kys(contexts: &[(&str, Vec<(&str, u64, &str)>)]) -> String {
    let mut s = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<PremiereData Version=\"3\"><shortcuts Version=\"5\"><platform>macintosh</platform>");
    for (ctx, items) in contexts {
        s.push_str(&format!("<context.{ctx} Version=\"1\"><itemcount>{}</itemcount>", items.len()));
        for (i, (c, k, m)) in items.iter().enumerate() {
            s.push_str(&kys_item(i, c, *k, m));
        }
        s.push_str(&format!("</context.{ctx}>"));
    }
    s.push_str("</shortcuts></PremiereData>\n");
    s
}

/// A typed character's key code (`0x8000_0000 | ASCII`).
const fn ch(c: u8) -> u64 {
    0x8000_0000 | c as u64
}

/// A small custom Premiere preset, written for these tests.
fn sample_kys() -> String {
    kys(&[
        (
            "global",
            vec![
                ("cmd.sequence.razorateditline", ch(b'K'), "cmd"),
                ("cmd.edit.rippledelete", ch(b'T'), ""),
                ("cmd.tlnav.select.clip.at.playhead", ch(b'D'), ""),
                ("cmd.sequence.rippletrimpreviousedittoplayhead", ch(b'Q'), ""),
                ("cmd.file.new.adjustmentlayer", ch(b'Y'), "cmd"),
                ("cmd.clip.settoframesize", ch(b'S'), "opt"),
                ("cmd.edit.duplicate", ch(b'/'), "cmd shift"),
                ("cmd.transport.toggleplay", 1, ""),
                ("cmd.tlnav.prev.edit", 44, ""),
                ("cmd.transport.step.back.five", 42, "shift"),
                ("cmd.sequence.decreaseclipvolume", ch(b'['), ""),
                ("cmd.toggle.maximize.frame", ch(b'`'), ""),
                ("cmd.help.contents", 7, ""),
                ("cmd.edit.clear", 35, ""),
                ("uif.window.Timelines", ch(b'3'), "shift"),
                ("cmd.tlnav.select.nearest.edit.as.ripple.out", ch(b'T'), "shift"),
                // no key: Snap loses its S
                ("cmd.sequence.snap", 0, ""),
                // Premiere-only
                ("cmd.titler.menu.templates", ch(b'J'), "cmd"),
                ("cmd.file.capture", 11, ""),
                ("cmd.mystery", 999, "cmd"),
            ],
        ),
        ("timeline", vec![("cmd.timeline.paste.to.same.track", ch(b'V'), "cmd"), ("cmd.edit.clear", 2, "")]),
        ("project", vec![("cmd.edit.clear", 2, ""), ("cmd.file.newbinfromselection", ch(b'B'), "shift")]),
        ("prproduction", vec![("cmd.prproductionfolder.makeacopy", ch(b'/'), "cmd shift")]),
    ])
}

#[test]
fn kys_keys_decode_from_premiere_codes() {
    use crate::premiere_kys::decode_key;
    assert_eq!(decode_key(2147483723), Some("K"), "0x8000_0000 | 'K'");
    assert_eq!(decode_key(ch(b'k')), Some("K"));
    assert_eq!(decode_key(ch(b'/')), Some("/"));
    assert_eq!(decode_key(ch(b'`')), Some("`"));
    assert_eq!(decode_key(ch(b'\\')), Some("\\"));
    assert_eq!(decode_key(1), Some("Space"));
    assert_eq!(decode_key(2), Some("Backspace"));
    assert_eq!(decode_key(4), Some("Enter"));
    assert_eq!(decode_key(7), Some("F1"));
    assert_eq!(decode_key(18), Some("F12"));
    assert_eq!(decode_key(35), Some("Delete"));
    assert_eq!((decode_key(42), decode_key(43), decode_key(44), decode_key(45)), (Some("Left"), Some("Right"), Some("Up"), Some("Down")));
    for bad in [0, 999, ch(b' '), ch(0x7f), 0x8000_0000 | 0x1F600, u64::MAX, 0xFFFF_FFFF] {
        assert_eq!(decode_key(bad), None, "{bad}");
    }
}

#[test]
fn kys_import_maps_commands_panels_and_keys() {
    let dir = temp_dir("kys");
    let path = dir.join("[Custom 2].kys");
    std::fs::write(&path, sample_kys()).unwrap();
    let mut s = Session::default();
    with_ui_commands(&mut s);
    s.shortcuts.register_external(vec![
        CommandInfo::new("playback.toggle", "Play/Stop", &[], Some("Space")),
        CommandInfo::new("tool.type", "Type Tool", &[], Some("T")),
        CommandInfo::new("window.panel.Timeline", "Timeline", &["Window"], Some("Shift+3")),
        CommandInfo::new("window.maximizeFrameUnderCursor", "Maximize Frame", &[], None),
        CommandInfo::new("help.filmcraftHelp", "FilmCraft Help", &[], None),
    ]);
    s.shortcuts.set_dir(&dir);
    // a dry run reports without changing anything
    let before = s.shortcuts.preset.clone();
    let r = s.execute("shortcuts.import", json!({"path": path.to_string_lossy(), "dryRun": true})).unwrap();
    assert_eq!(r["name"], json!("Premiere – Custom 2"));
    assert_eq!((r["saved"].clone(), r["activated"].clone()), (json!(false), json!(false)));
    assert_eq!(s.shortcuts.preset, before);
    // the real import
    let r = s.execute("shortcuts.import", json!({"path": path.to_string_lossy(), "name": "Pat – Premiere Custom 2"})).unwrap();
    assert_eq!(r["format"], json!("premiere-kys"));
    assert_eq!(r["total"], json!(24), "{r}");
    assert_eq!(r["mapped"], json!(20), "{r}");
    let unmapped: Vec<(String, String, String)> = r["unmapped"]
        .as_array()
        .unwrap()
        .iter()
        .map(|u| (u["command"].as_str().unwrap().to_string(), u["keys"].as_str().unwrap().to_string(), u["reason"].as_str().unwrap().to_string()))
        .collect();
    assert!(unmapped.contains(&("cmd.titler.menu.templates".into(), "Cmd+J".into(), "no FilmCraft command".into())), "{unmapped:?}");
    assert!(unmapped.contains(&("cmd.file.capture".into(), "F5".into(), "no FilmCraft command".into())));
    assert!(unmapped.contains(&("cmd.mystery".into(), "key code 999".into(), "no FilmCraft command".into())));
    assert!(unmapped.iter().any(|u| u.0 == "cmd.prproductionfolder.makeacopy" && u.2 == "panel not in FilmCraft"));
    assert_eq!(r["activated"], json!(true));
    assert_eq!(s.shortcuts.preset, "Pat – Premiere Custom 2");
    assert!(!s.shortcuts.modified);
    assert_eq!(r["presets"]["custom"], json!(["Pat – Premiere Custom 2"]));
    let sc = &s.shortcuts;
    let res = |k: &str, p: Option<&str>| sc.resolve(&Chord::parse(k).unwrap(), p, Platform::Mac).map(|b| b.command.clone());
    assert_eq!(res("T", None).as_deref(), Some("edit.rippleDelete"), "Pat's T");
    assert!(sc.for_command("tool.type").is_empty(), "T was taken from the Type tool");
    assert_eq!(res("Cmd+K", None).as_deref(), Some("sequence.addEdit"));
    assert_eq!(res("D", None).as_deref(), Some("timeline.selectClipAtPlayhead"));
    assert_eq!(res("Q", None).as_deref(), Some("trim.ripplePrevious"));
    assert_eq!(res("Cmd+Y", None).as_deref(), Some("file.newAdjustmentLayer"));
    assert_eq!(res("Alt+S", None).as_deref(), Some("clip.fitToFrame"));
    assert_eq!(res("Cmd+Shift+/", None).as_deref(), Some("edit.duplicate"));
    assert_eq!(res("Space", None).as_deref(), Some("playback.toggle"));
    assert_eq!(res("Up", None).as_deref(), Some("playhead.prevEdit"));
    assert_eq!(res("Shift+Left", None).as_deref(), Some("playhead.stepBack5"));
    assert_eq!(res("[", None).as_deref(), Some("clip.volumeDown"));
    assert_eq!(res("`", None).as_deref(), Some("window.maximizeFrameUnderCursor"));
    assert_eq!(res("F1", None).as_deref(), Some("help.filmcraftHelp"));
    assert_eq!(res("Shift+3", None).as_deref(), Some("window.panel.Timeline"));
    assert_eq!(res("Shift+T", None).as_deref(), Some("trim.selectNearestRippleOut"));
    assert!(sc.for_command("sequence.snap").is_empty(), "the file leaves Snap without a key");
    // panel contexts
    assert_eq!(res("Cmd+V", Some("Timeline")).as_deref(), Some("edit.paste"));
    assert_eq!(res("Backspace", Some("Timeline")).as_deref(), Some("edit.clear"));
    assert_eq!(res("Backspace", Some("Project")).as_deref(), Some("project.delete"), "Clear in the Project panel deletes items");
    assert_eq!(res("Shift+B", Some("Project")).as_deref(), Some("file.newBinFromSelection"));
    // commands the file doesn't mention keep their Premiere Pro Compatible keys
    assert_eq!(sc.primary("edit.undo").as_deref(), Some("Cmd+Z"));
    assert_eq!(sc.primary("markers.markIn").as_deref(), Some("I"));
    assert!(sc.conflicts(Platform::Mac).is_empty(), "{:?}", sc.conflicts(Platform::Mac));
    // the preset is a custom preset like any other: switch away and back
    s.execute("shortcuts.loadPreset", json!({"name": "FilmCraft Default"})).unwrap();
    assert_eq!(s.shortcuts.primary("tool.type").as_deref(), Some("T"));
    s.execute("shortcuts.loadPreset", json!({"name": "Pat – Premiere Custom 2"})).unwrap();
    assert_eq!(s.shortcuts.resolve(&Chord::parse("T").unwrap(), None, Platform::Mac).map(|b| b.command.as_str()), Some("edit.rippleDelete"));
    // default name, and built-in names are refused
    let r = s.execute("shortcuts.import", json!({"path": path.to_string_lossy(), "activate": false})).unwrap();
    assert_eq!((r["name"].clone(), r["activated"].clone()), (json!("Premiere – Custom 2"), json!(false)));
    assert!(s.execute("shortcuts.import", json!({"path": path.to_string_lossy(), "name": "Premiere Pro Compatible"})).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn kys_windows_files_use_ctrl_as_the_primary_modifier() {
    let text = kys(&[("global", vec![("cmd.sequence.razorateditline", ch(b'K'), "ctrl")])]).replace("macintosh", "windows");
    let f = crate::premiere_kys::parse(&text).unwrap();
    assert_eq!(f.platform, Platform::Windows);
    assert_eq!(f.items[0].chord.as_ref().map(Chord::canonical).as_deref(), Some("Cmd+K"));
}

/// Hostile files: errors, never a panic (AGENTS.md §0.8).
#[test]
fn kys_hostile_files_fail_cleanly() {
    let dir = temp_dir("kys-hostile");
    let mut s = Session::default();
    s.shortcuts.set_dir(&dir);
    let cases: Vec<Vec<u8>> = vec![
        b"<PremiereData".to_vec(),
        b"<PremiereData Version=\"3\"></PremiereData>".to_vec(),
        b"<PremiereData><shortcuts><context.global><item.0><virtualkey>-5</virtualkey><commandname>cmd.edit.undo</commandname></item.0></context.global></shortcuts></PremiereData>".to_vec(),
        b"<PremiereData><shortcuts><context.global><item.0><virtualkey>99999999999999999999999</virtualkey><commandname></commandname></item.0></context.global></shortcuts></PremiereData>".to_vec(),
        b"<!DOCTYPE x [<!ENTITY a \"aaaa\">]><PremiereData>&a;</PremiereData>".to_vec(),
        vec![0xff, 0xfe, b'<', 0, b'P', 0],
    ];
    for (i, bytes) in cases.iter().enumerate() {
        let p = dir.join(format!("bad{i}.kys"));
        std::fs::write(&p, bytes).unwrap();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("shortcuts.import", json!({"path": p.to_string_lossy()}))));
        assert!(r.is_ok(), "case {i} panicked");
    }
    assert!(s.execute("shortcuts.import", json!({"path": dir.join("missing.kys").to_string_lossy()})).is_err());
    // mutation fuzz: truncations and byte flips of a real-shaped file
    let good = sample_kys().into_bytes();
    let mut seed = 0x2545_f491_u64;
    for n in 0..300 {
        let mut b = good.clone();
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        if n % 3 == 0 {
            b.truncate((seed as usize) % b.len());
        } else {
            let at = (seed as usize) % b.len();
            b[at] ^= (seed >> 32) as u8 | 1;
        }
        let r = std::panic::catch_unwind(|| crate::premiere_kys::parse(&String::from_utf8_lossy(&b)).map(|f| crate::premiere_kys::convert(&f, |_| true)));
        assert!(r.is_ok(), "mutation {n} panicked");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
