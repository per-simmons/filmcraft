//! Project panel Copy / Cut / Paste and Edit ▸ Duplicate (`project_clipboard`).

use std::collections::BTreeSet;

use filmcraft_project::{BinEntry, BinId, ClipId, ItemId, ItemKind};
use serde_json::{Value, json};

use crate::Session;

fn demo() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s
}

fn named(s: &Session, name: &str) -> ItemId {
    s.project.items.values().find(|i| i.name == name).map(|i| i.id).unwrap_or_else(|| panic!("no item {name}"))
}

fn bin_named(s: &Session, name: &str) -> BinId {
    s.project.root.children.iter().find_map(|c| if let BinEntry::Bin(b) = c { (b.name == name).then_some(b.id) } else { None }).unwrap()
}

fn ids(v: &Value) -> Vec<u64> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
}

/// Every clip id of every sequence in the project.
fn all_clip_ids(s: &Session) -> Vec<ClipId> {
    s.project.sequences().filter_map(|i| i.as_sequence()).flat_map(|q| q.all_tracks().flat_map(|t| t.items.iter().map(|c| c.id)).collect::<Vec<_>>()).collect()
}

#[test]
fn duplicate_places_a_copy_below_the_original_without_asking() {
    let mut s = demo();
    let main = named(&s, "Main Edit");
    let seqs = bin_named(&s, "Sequences");
    s.execute("project.select", json!({"items": [main.0]})).unwrap();
    let r = s.execute("edit.duplicate", json!({})).unwrap();
    let copy = ItemId(r["items"][0].as_u64().unwrap());
    assert_eq!(r["bin"], json!(seqs.0), "{r}");
    assert_eq!(r["copies"][0]["name"], json!("Main Edit Copy"));
    assert_eq!(s.project.item(copy).unwrap().name, "Main Edit Copy");
    assert_eq!(s.project.bin_of(copy), Some(seqs), "same bin as the original");
    let order: Vec<ItemId> =
        s.project.root.find_bin(seqs).unwrap().children.iter().filter_map(|c| if let BinEntry::Item(i) = c { Some(*i) } else { None }).collect();
    let at = order.iter().position(|i| *i == main).unwrap();
    assert_eq!(order.get(at + 1), Some(&copy), "right below the original: {order:?}");
    assert_eq!(s.state.project_selection, vec![copy], "the copy is selected");
    // an independent copy: same edit, fresh ids everywhere
    let (a, b) = (s.project.sequence(main).unwrap().clone(), s.project.sequence(copy).unwrap().clone());
    assert_eq!(a.video_tracks[0].items.len(), b.video_tracks[0].items.len());
    assert!(!b.video_tracks[0].items.is_empty());
    for (x, y) in a.all_tracks().flat_map(|t| t.items.iter()).zip(b.all_tracks().flat_map(|t| t.items.iter())) {
        assert_eq!((x.item, x.start, x.duration, x.source_in), (y.item, y.start, y.duration, y.source_in));
        assert_ne!(x.id, y.id);
        assert_eq!(x.link.is_some(), y.link.is_some());
        if x.link.is_some() {
            assert_ne!(x.link, y.link, "the copy has its own link numbers");
        }
    }
    let clip_ids = all_clip_ids(&s);
    assert_eq!(clip_ids.len(), clip_ids.iter().collect::<BTreeSet<_>>().len(), "clip ids stay unique in the project");
    // linked pairs stay paired in the copy
    let v0 = &b.video_tracks[0].items[0];
    assert!(b.audio_tracks[0].items.iter().any(|a| a.link == v0.link));
    // editing the copy leaves the original alone
    s.execute("sequence.open", json!({"item": copy.0})).unwrap();
    s.execute("edit.selectAll", json!({})).unwrap();
    s.execute("edit.clear", json!({})).unwrap();
    assert!(s.project.sequence(copy).unwrap().video_tracks[0].items.is_empty());
    assert_eq!(s.project.sequence(main).unwrap().video_tracks[0].items.len(), a.video_tracks[0].items.len());
    // a second copy takes the next free name
    s.execute("project.select", json!({"items": [main.0]})).unwrap();
    let r = s.execute("edit.duplicate", json!({})).unwrap();
    assert_eq!(r["copies"][0]["name"], json!("Main Edit Copy 01"));
    // clips duplicate in their bin; undo takes the copy away
    let clip = named(&s, "Ocean_Sunset.mp4");
    let bin = s.project.bin_of(clip);
    let r = s.execute("edit.duplicate", json!({"items": [clip.0]})).unwrap();
    let c = ItemId(r["items"][0].as_u64().unwrap());
    assert_eq!((s.project.item(c).unwrap().name.as_str(), s.project.bin_of(c)), ("Ocean_Sunset.mp4 Copy", bin));
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.project.item(c).is_none());
    // bins alone can't be duplicated; nothing selected is an error
    assert!(s.execute("edit.duplicate", json!({"items": [bin.unwrap().0]})).is_err());
    s.execute("project.select", json!({"items": []})).unwrap();
    assert!(s.execute("edit.duplicate", json!({})).is_err());
}

#[test]
fn copy_and_paste_into_a_bin() {
    let mut s = demo();
    let main = named(&s, "Main Edit");
    let clip = named(&s, "Ocean_Sunset.mp4");
    let gfx = bin_named(&s, "Graphics");
    let n = s.project.items.len();
    s.execute("project.select", json!({"items": [main.0, clip.0]})).unwrap();
    let r = s.execute("project.copy", json!({})).unwrap();
    assert_eq!(ids(&r["items"]), vec![main.0, clip.0]);
    assert_eq!(s.project.items.len(), n, "copying changes nothing");
    let r = s.execute("project.paste", json!({"bin": gfx.0})).unwrap();
    let made = ids(&r["items"]);
    assert_eq!((made.len(), r["bin"].clone()), (2, json!(gfx.0)), "{r}");
    let names: Vec<String> = made.iter().map(|i| s.project.item(ItemId(*i)).unwrap().name.clone()).collect();
    assert_eq!(names, vec!["Main Edit Copy".to_string(), "Ocean_Sunset.mp4".to_string()], "sequences get “Copy”, clips keep their name");
    assert!(made.iter().all(|i| s.project.bin_of(ItemId(*i)) == Some(gfx)));
    assert_eq!(s.state.project_selection, made.iter().map(|i| ItemId(*i)).collect::<Vec<_>>());
    // paste again without a bin: into the bin of the selection (the pasted items, in Graphics)
    let r = s.execute("project.paste", json!({})).unwrap();
    assert_eq!(r["bin"], json!(gfx.0));
    assert_eq!(s.project.item(ItemId(ids(&r["items"])[0])).unwrap().name, "Main Edit Copy 01");
    // with a bin selected, paste goes into it
    s.execute("project.select", json!({"items": [bin_named(&s, "Audio").0]})).unwrap();
    let r = s.execute("project.paste", json!({})).unwrap();
    assert_eq!(r["bin"], json!(bin_named(&s, "Audio").0));
    // nothing selected: the top level
    s.execute("project.select", json!({"items": []})).unwrap();
    let r = s.execute("project.paste", json!({})).unwrap();
    assert_eq!(r["bin"], json!(s.project.root.id.0));
    assert_eq!(s.project.items.len(), n + 8);
    // one undo per paste
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(s.project.items.len(), n + 6);
}

#[test]
fn cut_and_paste_moves_items_and_bins() {
    let mut s = demo();
    let clip = named(&s, "Ocean_Sunset.mp4");
    let footage = s.project.bin_of(clip).unwrap();
    let seqs = bin_named(&s, "Sequences");
    let n = s.project.items.len();
    s.execute("project.cut", json!({"items": [clip.0]})).unwrap();
    assert_eq!(s.project.bin_of(clip), Some(footage), "cut alone moves nothing");
    let r = s.execute("project.paste", json!({"bin": seqs.0})).unwrap();
    assert_eq!((ids(&r["items"]), r["moved"].clone()), (vec![clip.0], json!(true)));
    assert_eq!(s.project.bin_of(clip), Some(seqs));
    assert_eq!(s.project.items.len(), n, "moved, not copied");
    assert!(s.execute("project.paste", json!({})).is_err(), "a cut pastes once");
    // a bin moves with its contents; never into itself
    let gfx = bin_named(&s, "Graphics");
    s.execute("project.cut", json!({"items": [gfx.0]})).unwrap();
    s.execute("project.paste", json!({"bin": seqs.0})).unwrap();
    assert_eq!(s.project.parent_bin(gfx), Some(seqs));
    s.execute("project.cut", json!({"items": [seqs.0]})).unwrap();
    assert!(s.execute("project.paste", json!({"bin": gfx.0})).is_err(), "a bin can't go inside itself");
    assert_eq!(s.project.parent_bin(seqs), Some(s.project.root.id));
    // copied bins are deep copies
    let audio = bin_named(&s, "Audio");
    let inside = s.project.root.find_bin(audio).unwrap().children.len();
    s.execute("project.copy", json!({"items": [audio.0]})).unwrap();
    let r = s.execute("project.paste", json!({"bin": s.project.root.id.0})).unwrap();
    let nb = BinId(ids(&r["bins"])[0]);
    let b = s.project.root.find_bin(nb).unwrap();
    assert_eq!((b.name.as_str(), b.children.len()), ("Audio", inside));
    let mut orig = Vec::new();
    s.project.root.find_bin(audio).unwrap().all_items(&mut orig);
    let mut copies = Vec::new();
    b.all_items(&mut copies);
    assert!(copies.iter().all(|c| !orig.contains(c)), "new items, not the same ones");
}

#[test]
fn timeline_copy_paste_is_separate_and_params_are_checked() {
    let mut s = demo();
    // the timeline clipboard is untouched by the Project panel's
    let first = s.active_sequence().unwrap().video_tracks[0].items[0].id;
    s.state.selection = vec![first];
    s.execute("edit.copy", json!({})).unwrap();
    let main = named(&s, "Main Edit");
    s.execute("project.copy", json!({"items": [main.0]})).unwrap();
    assert!(!s.state.clipboard.is_empty());
    let clips = s.active_sequence().unwrap().all_tracks().map(|t| t.items.len()).sum::<usize>();
    s.execute("edit.paste", json!({})).unwrap();
    assert!(s.active_sequence().unwrap().all_tracks().map(|t| t.items.len()).sum::<usize>() > clips, "Edit ▸ Paste still pastes clips");
    // hostile parameters
    assert!(s.execute("project.copy", json!({"items": [u64::MAX, 0, 999_999]})).is_err());
    assert!(s.execute("project.copy", json!({"items": [s.project.root.id.0]})).is_err(), "the root isn't an item");
    assert!(s.execute("project.copy", json!({"items": "x"})).is_err());
    assert!(s.execute("project.paste", json!({"bin": 999_999})).is_err());
    assert!(s.execute("project.paste", json!({"bin": "Footage"})).is_err());
    assert!(s.execute("project.paste", json!({"bin": -1})).is_err());
    assert!(s.execute("project.paste", json!({"bin": main.0})).is_err(), "an item isn't a bin");
    // the copied item went away before the paste
    s.execute("project.delete", json!({"items": [main.0]})).unwrap();
    assert!(s.execute("project.paste", json!({})).is_err());
    let _ = ItemKind::Graphic { width: 1, height: 1, rate: filmcraft_time::FrameRate::FPS_24 };
}
