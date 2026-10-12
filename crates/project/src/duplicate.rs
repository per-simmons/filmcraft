//! Copies of project items and bins (Edit ▸ Duplicate, Copy / Paste in the Project panel), and
//! moving bins.
//!
//! A copy is independent of its original: a duplicated sequence gets fresh track, clip,
//! transition, caption and marker ids (clip ids are unique across the project) and its own
//! link / group numbers, so editing either never touches the other. Media a copy uses is shared.

use std::collections::BTreeMap;

use crate::{Bin, BinEntry, BinId, ClipId, ItemId, ItemKind, MarkerId, Project, TrackId, TransitionId};

/// Bins nested deeper than this are not copied (project files are untrusted).
const MAX_BIN_DEPTH: usize = 64;

impl Project {
    /// The bin holding an item (the root for an item at the top level); None for an unknown id.
    pub fn bin_of(&self, id: ItemId) -> Option<BinId> {
        self.root.parent_of(id)
    }

    /// The bin holding a bin (None for the root or an unknown bin).
    pub fn parent_bin(&self, bin: BinId) -> Option<BinId> {
        fn walk(b: &Bin, id: BinId, depth: usize) -> Option<BinId> {
            if depth > MAX_BIN_DEPTH {
                return None;
            }
            b.children.iter().find_map(|c| match c {
                BinEntry::Bin(sub) if sub.id == id => Some(b.id),
                BinEntry::Bin(sub) => walk(sub, id, depth + 1),
                BinEntry::Item(_) => None,
            })
        }
        walk(&self.root, bin, 0)
    }

    /// Premiere's name for a copy: `<name> Copy`, then `<name> Copy 01`, `Copy 02`… while the name
    /// is taken by another item.
    pub fn copy_name(&self, name: &str) -> String {
        let taken = |n: &str| self.items.values().any(|i| i.name == n);
        let first = format!("{name} Copy");
        if !taken(&first) {
            return first;
        }
        (1..1000).map(|i| format!("{name} Copy {i:02}")).find(|n| !taken(n)).unwrap_or(first)
    }

    /// A copy of item `id` named `name`, put into `bin` (the original's bin when None) right after
    /// the original when it is in that bin, else at the end. None when `id` doesn't exist.
    pub fn duplicate_item(&mut self, id: ItemId, name: &str, bin: Option<BinId>) -> Option<ItemId> {
        let mut it = self.items.get(&id)?.clone();
        let nid = ItemId(self.alloc_id());
        it.id = nid;
        it.name = name.to_string();
        it.created = nid.0;
        self.fresh_ids(&mut it.kind);
        self.items.insert(nid, it);
        if let Some(t) = self.transcripts.get(&id).cloned() {
            self.transcripts.insert(nid, t);
        }
        if let Some(g) = self.source_graphics.get(&id).cloned() {
            self.source_graphics.insert(nid, g);
        }
        let target = bin.filter(|b| self.root.find_bin(*b).is_some()).or_else(|| self.bin_of(id)).unwrap_or(self.root.id);
        let dest = match self.root.find_bin_mut(target) {
            Some(b) => b,
            None => &mut self.root,
        };
        match dest.children.iter().position(|c| matches!(c, BinEntry::Item(i) if *i == id)) {
            Some(at) => dest.children.insert(at + 1, BinEntry::Item(nid)),
            None => dest.children.push(BinEntry::Item(nid)),
        }
        Some(nid)
    }

    /// A copy of bin `id` and everything in it (items keep their names) inside `parent`. Returns
    /// the new bin and the new item ids. None when the bin is unknown, is the root, or `parent`
    /// lies inside it.
    pub fn duplicate_bin(&mut self, id: BinId, parent: BinId) -> Option<(BinId, Vec<ItemId>)> {
        if id == self.root.id {
            return None;
        }
        let src = self.root.find_bin(id)?.clone();
        if src.find_bin(parent).is_some() || self.root.find_bin(parent).is_none() {
            return None;
        }
        let mut items = Vec::new();
        let copy = self.copy_bin(&src, &mut items, 0)?;
        let nid = copy.id;
        self.root.find_bin_mut(parent)?.children.push(BinEntry::Bin(copy));
        Some((nid, items))
    }

    fn copy_bin(&mut self, src: &Bin, items: &mut Vec<ItemId>, depth: usize) -> Option<Bin> {
        if depth > MAX_BIN_DEPTH {
            return None;
        }
        let mut out = Bin { id: BinId(self.alloc_id()), name: src.name.clone(), children: Vec::new() };
        for c in &src.children {
            match c {
                BinEntry::Item(i) => {
                    let Some(it) = self.items.get(i).cloned() else { continue };
                    let nid = ItemId(self.alloc_id());
                    let mut it = it;
                    it.id = nid;
                    it.created = nid.0;
                    self.fresh_ids(&mut it.kind);
                    self.items.insert(nid, it);
                    if let Some(t) = self.transcripts.get(i).cloned() {
                        self.transcripts.insert(nid, t);
                    }
                    if let Some(g) = self.source_graphics.get(i).cloned() {
                        self.source_graphics.insert(nid, g);
                    }
                    out.children.push(BinEntry::Item(nid));
                    items.push(nid);
                }
                BinEntry::Bin(b) => {
                    if let Some(sub) = self.copy_bin(b, items, depth + 1) {
                        out.children.push(BinEntry::Bin(sub));
                    }
                }
            }
        }
        Some(out)
    }

    /// Move bin `id` (with its contents) into `to`. Returns false when it is already there; None
    /// when either bin is unknown, `id` is the root, or `to` lies inside `id`.
    pub fn move_bin(&mut self, id: BinId, to: BinId) -> Option<bool> {
        if id == self.root.id || self.root.find_bin(to).is_none() || self.root.find_bin(id)?.find_bin(to).is_some() {
            return None;
        }
        if self.parent_bin(id) == Some(to) {
            return Some(false);
        }
        let b = self.root.remove_bin(id)?;
        match self.root.find_bin_mut(to) {
            Some(t) => t.children.push(BinEntry::Bin(b)),
            None => self.root.children.push(BinEntry::Bin(b)),
        }
        Some(true)
    }

    /// Give everything inside a copied item fresh ids.
    fn fresh_ids(&mut self, kind: &mut ItemKind) {
        match kind {
            ItemKind::Sequence(seq) => {
                for m in &mut seq.markers {
                    m.id = MarkerId(self.alloc_id());
                }
                let mut links: BTreeMap<u64, u64> = BTreeMap::new();
                let mut groups: BTreeMap<u64, u64> = BTreeMap::new();
                let tracks = seq.video_tracks.iter_mut().chain(seq.audio_tracks.iter_mut()).chain(seq.submix_tracks.iter_mut());
                for t in tracks {
                    t.id = TrackId(self.alloc_id());
                    let mut clips: BTreeMap<ClipId, ClipId> = BTreeMap::new();
                    for ti in &mut t.items {
                        let n = ClipId(self.alloc_id());
                        clips.insert(ti.id, n);
                        ti.id = n;
                        ti.link = ti.link.map(|l| renumber(self, &mut links, l));
                        ti.group = ti.group.map(|g| renumber(self, &mut groups, g));
                        for m in &mut ti.markers {
                            m.id = MarkerId(self.alloc_id());
                        }
                    }
                    for tr in &mut t.transitions {
                        tr.id = TransitionId(self.alloc_id());
                        tr.from = tr.from.and_then(|c| clips.get(&c).copied());
                        tr.to = tr.to.and_then(|c| clips.get(&c).copied());
                    }
                }
                for ct in &mut seq.caption_tracks {
                    ct.id = TrackId(self.alloc_id());
                    for c in &mut ct.captions {
                        c.id = ClipId(self.alloc_id());
                    }
                }
            }
            ItemKind::Media(m) => {
                for mk in &mut m.markers {
                    mk.id = MarkerId(self.alloc_id());
                }
            }
            ItemKind::Subclip { .. } | ItemKind::AdjustmentLayer { .. } | ItemKind::Graphic { .. } => {}
        }
    }
}

/// The copy's number for link / group `old` (one fresh number per original number).
fn renumber(p: &mut Project, map: &mut BTreeMap<u64, u64>, old: u64) -> u64 {
    if let Some(n) = map.get(&old) {
        return *n;
    }
    let n = p.alloc_id();
    map.insert(old, n);
    n
}
