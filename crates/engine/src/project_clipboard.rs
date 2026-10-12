//! Copy, Cut, Paste and Duplicate of Project panel items and bins, as in Premiere.
//!
//! - `project.copy {items?}` / `project.cut {items?}`: remember the items and bins (`items` takes
//!   item and bin ids; default: the Project panel selection). Cut moves them on the next paste; until then nothing changes.
//! - `project.paste {bin?}`: copies go into `bin` (default: the selected bin, else the bin of the
//!   first selected item, else the project's top level). Copied sequences are named
//!   `<name> Copy` (`Copy 01`… when taken); other items keep their names. Cut items move there.
//! - `edit.duplicate {items?}` (Edit ▸ Duplicate, ⌘⇧/): a `<name> Copy` of each item right
//!   below its original, in the same bin. No dialog.
//!
//! All of them select what they made and report it (`items`, `bins`, `bin`), so the Project panel
//! can scroll to it and an agent can use the new ids. The timeline's Copy / Paste are
//! `edit.copy` / `edit.paste`; the frontend sends Cmd+C / Cmd+X / Cmd+V to these commands while the
//! Project panel has focus.

use serde_json::{Value, json};

use filmcraft_project::{BinId, ItemId, ItemKind};

use crate::commands::{CommandSpec, bad, u64_p};
use crate::{Result, Session};

/// The ids a command acts on: `items` (or `item`) or the Project panel selection; each must be an
/// item or a bin (not the root).
fn targets(s: &Session, p: &Value, cmd: &str) -> Result<Vec<u64>> {
    let named: Option<Vec<u64>> =
        p.get("items").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect()).or_else(|| u64_p(p, "item").map(|i| vec![i]));
    let ids = named.unwrap_or_else(|| s.state.project_selection.iter().map(|i| i.0).collect());
    let root = s.project.root.id.0;
    let mut out: Vec<u64> = Vec::new();
    for id in ids {
        let ok = s.project.item(ItemId(id)).is_some() || (id != root && s.project.root.find_bin(BinId(id)).is_some());
        if ok && !out.contains(&id) {
            out.push(id);
        }
    }
    if out.is_empty() {
        return Err(bad(cmd, "select items or bins in the Project panel (or pass `items`)"));
    }
    Ok(out)
}

fn is_bin(s: &Session, id: u64) -> bool {
    s.project.item(ItemId(id)).is_none() && s.project.root.find_bin(BinId(id)).is_some()
}

fn copy(s: &mut Session, p: &Value, cut: bool) -> Result<Value> {
    let cmd = if cut { "project.cut" } else { "project.copy" };
    let ids = targets(s, p, cmd)?;
    s.state.project_clipboard = ids.clone();
    s.state.project_clipboard_cut = cut;
    Ok(json!({"items": ids, "cut": cut}))
}

/// Where a paste goes without a `bin`: the selected bin, else the bin of the first selected item.
fn default_bin(s: &Session) -> BinId {
    let sel = &s.state.project_selection;
    if let Some(b) = sel.iter().map(|i| BinId(i.0)).find(|b| is_bin(s, b.0)) {
        return b;
    }
    sel.iter().find_map(|i| s.project.bin_of(*i)).unwrap_or(s.project.root.id)
}

fn paste(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "project.paste";
    let clip = s.state.project_clipboard.clone();
    if clip.is_empty() {
        return Err(bad(cmd, "nothing to paste: copy or cut items in the Project panel first"));
    }
    let bin = match p.get("bin") {
        Some(Value::Null) | None => default_bin(s),
        Some(_) => {
            let b = BinId(p.get("bin").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "`bin` must be a bin id"))?);
            if s.project.root.find_bin(b).is_none() {
                return Err(bad(cmd, format!("no bin {}", b.0)));
            }
            b
        }
    };
    let cut = s.state.project_clipboard_cut;
    let mut made_items: Vec<u64> = Vec::new();
    let mut made_bins: Vec<u64> = Vec::new();
    let mut copies: Vec<(ItemId, ItemId)> = Vec::new();
    let mut skipped: Vec<u64> = Vec::new();
    s.edit(if cut { "Move" } else { "Paste" }, |pr, st| {
        for &id in &clip {
            let as_bin = pr.item(ItemId(id)).is_none() && pr.root.find_bin(BinId(id)).is_some();
            if cut {
                let moved = if as_bin { pr.move_bin(BinId(id), bin).is_some() } else { pr.move_to_bin(&[ItemId(id)], Some(bin)).is_some_and(|n| n > 0) };
                if moved {
                    if as_bin { made_bins.push(id) } else { made_items.push(id) }
                } else {
                    skipped.push(id);
                }
            } else if as_bin {
                match pr.duplicate_bin(BinId(id), bin) {
                    Some((b, _)) => made_bins.push(b.0),
                    None => skipped.push(id),
                }
            } else {
                let Some(it) = pr.item(ItemId(id)) else {
                    skipped.push(id);
                    continue;
                };
                let name = if matches!(it.kind, ItemKind::Sequence(_)) { pr.copy_name(&it.name) } else { it.name.clone() };
                match pr.duplicate_item(ItemId(id), &name, Some(bin)) {
                    Some(n) => {
                        made_items.push(n.0);
                        copies.push((ItemId(id), n));
                    }
                    None => skipped.push(id),
                }
            }
        }
        if made_items.is_empty() && made_bins.is_empty() {
            return Err(bad(cmd, "the copied items no longer exist"));
        }
        st.project_selection = made_items.iter().chain(&made_bins).map(|i| ItemId(*i)).collect();
        if cut {
            st.project_clipboard.clear();
            st.project_clipboard_cut = false;
        }
        Ok(())
    })?;
    share_media(s, &copies);
    Ok(json!({"items": made_items, "bins": made_bins, "bin": bin.0, "moved": cut, "skipped": skipped}))
}

/// Copies of media items use the original's already-open source (no second probe).
fn share_media(s: &Session, copies: &[(ItemId, ItemId)]) {
    for (from, to) in copies {
        if let Some(src) = s.media.cached(*from) {
            s.media.insert(*to, src);
        }
    }
}

/// Edit ▸ Duplicate: `<name> Copy` right below each selected item, in its bin.
pub(crate) fn duplicate(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "edit.duplicate";
    let ids: Vec<ItemId> = targets(s, p, cmd)?.into_iter().map(ItemId).filter(|i| s.project.item(*i).is_some()).collect();
    if ids.is_empty() {
        return Err(bad(cmd, "bins can't be duplicated; select items"));
    }
    let mut made: Vec<(ItemId, ItemId, BinId)> = Vec::new();
    s.edit("Duplicate", |pr, st| {
        for id in &ids {
            let Some(name) = pr.item(*id).map(|it| pr.copy_name(&it.name)) else { continue };
            let bin = pr.bin_of(*id).unwrap_or(pr.root.id);
            if let Some(n) = pr.duplicate_item(*id, &name, Some(bin)) {
                made.push((*id, n, bin));
            }
        }
        st.project_selection = made.iter().map(|m| m.1).collect();
        Ok(())
    })?;
    let pairs: Vec<(ItemId, ItemId)> = made.iter().map(|m| (m.0, m.1)).collect();
    share_media(s, &pairs);
    Ok(json!({
        "items": made.iter().map(|m| m.1.0).collect::<Vec<_>>(),
        "bin": made.first().map(|m| m.2.0),
        "copies": made.iter().map(|(o, n, b)| json!({"of": o.0, "item": n.0, "bin": b.0, "name": s.project.item(*n).map(|i| i.name.clone())})).collect::<Vec<_>>(),
    }))
}

fn has_project_clipboard(s: &Session) -> std::result::Result<(), String> {
    if s.state.project_clipboard.is_empty() { Err("copy or cut items in the Project panel first".into()) } else { Ok(()) }
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    let c = |id, label, params, enabled, run| CommandSpec { id, label, menu: &[], shortcut: None, params, enabled, run, journal: true };
    vec![
        c("project.copy", "Copy Project Items", r#"{"items":[id]?}"#, crate::commands::has_project_selection, |s, p| copy(s, p, false)),
        c("project.cut", "Cut Project Items", r#"{"items":[id]?}"#, crate::commands::has_project_selection, |s, p| copy(s, p, true)),
        c("project.paste", "Paste Project Items", r#"{"bin":binId?}"#, has_project_clipboard, paste),
    ]
}
