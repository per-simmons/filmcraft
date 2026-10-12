//! Retake detection on the sequence transcript (`transcript.findRepeats`, `transcript.cutRepeats`).
//!
//! A talking-head take is full of restarts: "So it's not only highly downloaded…" and then, a
//! breath later, "So it's not only highly downloaded, being one of the top applications…". An
//! agent reading the transcript misses them; this finds them deterministically.
//!
//! A hit is a pair of word runs: A (earlier) and B (later, starting within `windowSeconds` of A)
//! that open with the same words, compared normalised (lower case, outer punctuation stripped,
//! curly apostrophes straightened, contractions kept, filler words such as "um"/"uh" skipped):
//! - **exact**: an identical opening of at least `minWords` words;
//! - **near**: for runs of at least 5 words, one word substituted, dropped or added after an
//!   identical first word ("which I've reviewed in the past" / "which I've done in the past"),
//!   with at least 4 words shared and 2 shared after the difference.
//!
//! The suggested cut is the abandoned attempt: from A's first word up to (not including) B's first
//! word. Even when A completed the sentence and B is a re-read, A is the one cut (Pat's rule: keep
//! the LAST complete reading). Each A pairs with its nearest re-attempt, so three attempts give two
//! adjacent cuts.
//!
//! Skipped as false positives:
//! - openings made only of common words ("and then you can see"); an opening of fewer than 4
//!   words needs 2 content-bearing words, a longer one at least 1;
//! - deliberate emphasis: a phrase of at most 3 words repeated back to back ("we get it. we get
//!   it.", "going and going and going");
//! - pairs whose words in between dwarf what they share (more than `max(20, 4 × shared)` words
//!   between A's shared run and B): the same phrase used again later, not a retake;
//! - a hit whose cut would run into the later reading another hit keeps.

use std::ops::Range;

use serde_json::{Value, json};

use filmcraft_edit::transcript::{self as tx, DEFAULT_FILLERS, SeqWord};
use filmcraft_time::{TICKS_PER_SECOND, Tick};

use crate::commands::{CommandSpec, bad, bool_p, f64_p, u64_p};
use crate::transcript::sequence_words;
use crate::{Result, Session};

/// Settings of a search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepeatOptions {
    /// Shortest shared opening, in words (fillers not counted).
    pub min_words: usize,
    /// B must start within this long after A starts.
    pub window: Tick,
    /// Both runs must start in this stretch (sequence time).
    pub from: Option<Tick>,
    pub to: Option<Tick>,
}

impl Default for RepeatOptions {
    fn default() -> Self {
        RepeatOptions { min_words: 3, window: Tick::from_seconds_f64(45.0), from: None, to: None }
    }
}

/// One retake.
#[derive(Clone, Debug, PartialEq)]
pub struct Repeat {
    /// The abandoned attempt: words of the sequence transcript to cut (`from..to+1`).
    pub cut: Range<usize>,
    /// First word of the later reading (kept).
    pub later: usize,
    /// Word indices of A's shared opening (fillers skipped), and how many of them match B.
    pub shared: Vec<usize>,
    pub matched: usize,
    /// One word differs (substituted, dropped or added).
    pub near: bool,
    /// A ended a sentence before B began (B is a re-read of a complete attempt).
    pub complete: bool,
}

/// Longest gap allowed by the "in between dwarfs what they share" rule.
fn max_gap(matched: usize) -> usize {
    matched.saturating_mul(4).max(20)
}

/// Words so common that an opening made of them proves nothing.
const COMMON: &[&str] = &[
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "anyway",
    "are",
    "aren't",
    "as",
    "at",
    "back",
    "be",
    "because",
    "been",
    "before",
    "being",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "come",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "each",
    "even",
    "every",
    "for",
    "from",
    "get",
    "gets",
    "getting",
    "go",
    "goes",
    "going",
    "gonna",
    "got",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "here's",
    "him",
    "his",
    "how",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it'll",
    "it's",
    "its",
    "just",
    "kind",
    "know",
    "let",
    "let's",
    "like",
    "lot",
    "make",
    "me",
    "more",
    "most",
    "much",
    "my",
    "need",
    "no",
    "not",
    "now",
    "of",
    "off",
    "oh",
    "ok",
    "okay",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "pretty",
    "really",
    "right",
    "say",
    "see",
    "she",
    "so",
    "some",
    "something",
    "sort",
    "still",
    "such",
    "sure",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "them",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they're",
    "thing",
    "things",
    "think",
    "this",
    "those",
    "through",
    "to",
    "too",
    "up",
    "us",
    "very",
    "wanna",
    "want",
    "was",
    "way",
    "we",
    "we'll",
    "we're",
    "we've",
    "well",
    "were",
    "what",
    "what's",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "would",
    "yeah",
    "yes",
    "you",
    "you'll",
    "you're",
    "you've",
    "your",
];

fn is_common(w: &str) -> bool {
    COMMON.contains(&w)
}

/// A word as compared: lower case, outer punctuation stripped, contractions kept.
pub fn normalize(text: &str) -> String {
    let t: String = text.chars().map(|c| if c == '\u{2019}' || c == '\u{2018}' { '\'' } else { c }).collect();
    filmcraft_project::transcript::normalize_word(&t)
}

/// Whether the shared words carry enough meaning to be a retake rather than stock phrasing.
fn meaningful(shared: &[&str]) -> bool {
    let content = shared.iter().filter(|w| !is_common(w)).count();
    if shared.len() < 4 { content >= 2 } else { content >= 1 }
}

/// Words `i..j+len` repeat with a period of at most 3 words: deliberate emphasis.
fn emphasis(toks: &[(usize, String)], i: usize, end: usize) -> bool {
    let Some(span) = toks.get(i..end) else { return false };
    (1..=3usize).any(|q| span.len() >= 2 * q && span.iter().zip(span.iter().skip(q)).all(|(a, b)| a.1 == b.1))
}

/// How A (from token `i`) and B (from token `j`) share their opening: (A tokens used, matched
/// tokens, near), or None when they don't qualify. A may not run into B.
fn compare(toks: &[(usize, String)], i: usize, j: usize, min_words: usize) -> Option<(usize, usize, bool)> {
    let same = |a: usize, b: usize| a < j && matches!((toks.get(a), toks.get(b)), (Some(x), Some(y)) if x.1 == y.1);
    let run = |a: usize, b: usize| (0..).take_while(|k| same(a + k, b + k)).count();
    let exact = run(i, j);
    if exact >= min_words {
        return Some((exact, exact, false));
    }
    // near: an identical first word, one word different, then at least 2 more shared
    if exact == 0 {
        return None;
    }
    let (a, b) = (i + exact, j + exact);
    let options = [
        (1, run(a + 1, b + 1)), // substituted
        (1, run(a + 1, b)),     // A has a word B dropped
        (0, run(a, b + 1)),     // B added a word
    ];
    let (extra_a, tail) = options.into_iter().filter(|(_, t)| *t >= 2).max_by_key(|(e, t)| (*t, *e))?;
    let matched = exact + tail;
    let used = matched + extra_a;
    (used >= 5 && matched >= min_words.max(4)).then_some((used, matched, true))
}

/// Every retake in `words`, sorted by time.
pub fn find_repeats(words: &[SeqWord], o: &RepeatOptions) -> Vec<Repeat> {
    let min_words = o.min_words.max(2);
    let toks: Vec<(usize, String)> =
        words.iter().enumerate().map(|(i, w)| (i, normalize(&w.text))).filter(|(_, n)| !n.is_empty() && !DEFAULT_FILLERS.contains(&n.as_str())).collect();
    let start = |k: usize| toks.get(k).and_then(|t| words.get(t.0)).map(|w| w.start).unwrap_or_default();
    let mut out: Vec<Repeat> = Vec::new();
    // both runs start inside from..=to; the word before A may be outside (it decides an opening)
    let inside = |k: usize| o.from.is_none_or(|f| start(k) >= f) && o.to.is_none_or(|t| start(k) <= t);
    for i in 0..toks.len() {
        if !inside(i) {
            continue;
        }
        let a0 = start(i);
        for j in i + 1..toks.len() {
            if start(j) - a0 > o.window || !inside(j) {
                break;
            }
            // only openings: the words before A and B differ
            if i > 0 && toks.get(i - 1).map(|t| &t.1) == toks.get(j - 1).map(|t| &t.1) {
                continue;
            }
            let Some((used, matched, near)) = compare(&toks, i, j, min_words) else { continue };
            let shared: Vec<&str> = toks.get(i..i + used).unwrap_or_default().iter().map(|t| t.1.as_str()).collect();
            if !meaningful(&shared) || emphasis(&toks, i, j + matched) || j.saturating_sub(i + used) > max_gap(matched) {
                continue;
            }
            let (Some(a), Some(b)) = (toks.get(i), toks.get(j)) else { continue };
            let last = toks.get(j - 1).and_then(|t| words.get(t.0)).map(|w| w.text.trim_end_matches(['"', '\'', ')', '\u{201d}'])).unwrap_or("");
            out.push(Repeat {
                cut: a.0..b.0,
                later: b.0,
                shared: toks.get(i..i + used).unwrap_or_default().iter().map(|t| t.0).collect(),
                matched,
                near,
                complete: last.ends_with(['.', '!', '?']),
            });
            // the nearest re-attempt
            break;
        }
    }
    // a cut may not reach into a reading another hit keeps
    let mut kept: Vec<Repeat> = Vec::with_capacity(out.len());
    for r in out {
        if kept.iter().any(|k| k.cut.start < r.cut.start && r.cut.start < k.later && r.cut.end > k.later) {
            continue;
        }
        kept.push(r);
    }
    kept
}

/// Text of words `r`, at most `max` words (then "…").
fn text(words: &[SeqWord], r: Range<usize>, max: usize) -> String {
    let n = r.len();
    let mut s = words.get(r).unwrap_or_default().iter().take(max).map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
    if n > max {
        s.push_str(" …");
    }
    s
}

fn options(p: &Value, cmd: &str) -> Result<RepeatOptions> {
    let d = RepeatOptions::default();
    let secs = |k: &str| -> Result<Option<f64>> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(_) => match f64_p(p, k).filter(|v| v.is_finite() && *v >= 0.0) {
                Some(v) => Ok(Some(v.min(1.0e7))),
                None => Err(bad(cmd, format!("`{k}` must be a number of seconds ≥ 0"))),
            },
        }
    };
    let min_words = match p.get("minWords") {
        None | Some(Value::Null) => d.min_words,
        Some(_) => u64_p(p, "minWords")
            .filter(|n| (1..=1000).contains(n))
            .ok_or_else(|| bad(cmd, "`minWords` must be a whole number from 1 to 1000 (below 2 counts as 2)"))? as usize,
    };
    let window = secs("windowSeconds")?.filter(|w| *w > 0.0).map(|w| Tick::from_seconds_f64(w.min(3600.0))).unwrap_or(d.window);
    Ok(RepeatOptions { min_words: min_words.max(2), window, from: secs("from")?.map(Tick::from_seconds_f64), to: secs("to")?.map(Tick::from_seconds_f64) })
}

fn hits_json(s: &Session, words: &[SeqWord], hits: &[Repeat]) -> Vec<Value> {
    let rate = s.sequence_rate();
    let secs = |t: Tick| t.0 as f64 / TICKS_PER_SECOND as f64;
    hits.iter()
        .enumerate()
        .map(|(n, h)| {
            let to = h.cut.end.saturating_sub(1).max(h.cut.start);
            let r = tx::word_range(words, h.cut.start, to, rate);
            let at = |i: usize| words.get(i).map(|w| w.start).unwrap_or_default();
            let shared: Vec<&str> = h.shared.iter().filter_map(|&i| words.get(i)).map(|w| w.text.as_str()).collect();
            json!({
                "i": n,
                "kind": if h.near { "near" } else { "exact" },
                "sharedWords": h.matched,
                "shared": shared.join(" "),
                "earlier": {"from": h.cut.start, "to": to, "start": at(h.cut.start).0, "seconds": secs(at(h.cut.start)), "complete": h.complete, "text": text(words, h.cut.clone(), 40)},
                "later": {"from": h.later, "start": at(h.later).0, "seconds": secs(at(h.later)), "text": text(words, h.later..h.later.saturating_add(12).min(words.len()), 12)},
                "cut": {"from": h.cut.start, "to": to, "start": r.map(|r| r.start.0), "end": r.map(|r| r.end().0), "seconds": r.map(|r| secs(r.duration))},
            })
        })
        .collect()
}

/// `transcript.findRepeats`: every retake of the active sequence's transcript.
fn find(s: &mut Session, p: &Value) -> Result<Value> {
    let o = options(p, "transcript.findRepeats")?;
    let words = sequence_words(s);
    let hits = find_repeats(&words, &o);
    Ok(json!({"hits": hits_json(s, &words, &hits)}))
}

/// `transcript.cutRepeats`: ripple-delete every suggested cut (right to left, one undo step).
fn cut(s: &mut Session, p: &Value) -> Result<Value> {
    let o = options(p, "transcript.cutRepeats")?;
    let words = sequence_words(s);
    let hits = find_repeats(&words, &o);
    let report = hits_json(s, &words, &hits);
    let rate = s.sequence_rate();
    let ranges: Vec<_> = hits.iter().filter_map(|h| tx::word_range(&words, h.cut.start, h.cut.end.saturating_sub(1).max(h.cut.start), rate)).collect();
    if bool_p(p, "dryRun").unwrap_or(false) || ranges.is_empty() {
        return Ok(json!({"removed": 0, "dryRun": bool_p(p, "dryRun").unwrap_or(false), "hits": report}));
    }
    let n = ranges.len();
    let trs = s.project.transcripts.clone();
    let total = s.edit_sequence("Cut Repeats", |q, ctx, _| Ok(tx::ripple_delete_ranges(q, ranges, &trs, ctx)))?;
    Ok(json!({"removed": n, "ticks": total.0, "seconds": total.0 as f64 / TICKS_PER_SECOND as f64, "hits": report}))
}

fn has_transcript(s: &Session) -> std::result::Result<(), String> {
    crate::commands::has_seq(s)?;
    if sequence_words(s).is_empty() { Err("the sequence has no transcript (Transcribe first)".into()) } else { Ok(()) }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "transcript.findRepeats",
            label: "Find Retakes",
            menu: &[],
            shortcut: None,
            params: r#"{"minWords":n?=3,"windowSeconds":f?=45,"from":seconds?,"to":seconds?}"#,
            enabled: crate::commands::always,
            run: find,
            journal: false,
        },
        CommandSpec {
            id: "transcript.cutRepeats",
            label: "Cut Retakes",
            menu: &[],
            shortcut: None,
            params: r#"{"minWords":n?=3,"windowSeconds":f?=45,"from":seconds?,"to":seconds?,"dryRun":bool?}"#,
            enabled: has_transcript,
            run: cut,
            journal: true,
        },
    ]
}
