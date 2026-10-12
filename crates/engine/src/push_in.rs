//! `clip.pushIn`: an eased, motion-blurred Transform push-in in one command — the editor's usual
//! adjustment-layer push-in (New Adjustment Layer, place it over a range on a free track above
//! the footage, apply Transform, keyframe Scale (and Position, to reframe toward a third), Ease
//! In + Ease Out, Use Composition's Shutter Angle off, Shutter Angle on).
//!
//! It only uses the ordinary model: the result is a normal adjustment layer (or clip) with a
//! normal Transform effect and keyframes, so `effects.setParam`, `effects.setInterpolation`,
//! `effects.setKeyframe` and the Effect Controls panel edit it afterwards like any other. Run it
//! again on the same layer (`clip`) to add another move: it reuses the layer's Transform.

use filmcraft_geom::Vec2;
use filmcraft_project::{EffectInstance, Interpolation, ItemKind, Label, ParamValue, find_effect, resolve_auto_points};
use filmcraft_time::{Tick, TimeRange};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, clip_p, f64_p, has_seq, place_item, str_p, time_p, track_p};
use crate::{EngineError, Result, Session};

const CMD: &str = "clip.pushIn";

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Push-In…",
        menu: &[],
        shortcut: None,
        params: r#"{"clip":id?,"track":"V2"|id?,"startSeconds"|"startFrame"|"startTime"|"startTimecode":…?,"endSeconds"|"endFrame"|"endTime"|"endTimecode":…?,"moveStartSeconds"|…:?,"moveEndSeconds"|…:?,"moveSeconds":f64=0.75,"from":f64?,"to":f64=110,"position":[x,y]?,"fromPosition":[x,y]?,"anchor":[x,y]?,"ease":"bezier|easeIn|easeOut|linear|autoBezier|continuousBezier"="bezier","shutterAngle":0..360=120}"#,
        enabled: has_seq,
        run: push_in,
        journal: true,
    }]
}

fn interp(name: &str) -> Option<Interpolation> {
    Some(match name.to_ascii_lowercase().as_str() {
        "bezier" | "easeinout" | "ease" => Interpolation::Bezier,
        "easein" => Interpolation::EaseIn,
        "easeout" => Interpolation::EaseOut,
        "linear" => Interpolation::Linear,
        "autobezier" => Interpolation::AutoBezier,
        "continuousbezier" => Interpolation::ContinuousBezier,
        _ => return None,
    })
}

fn point_p(p: &Value, k: &str) -> Result<Option<Vec2>> {
    let Some(v) = p.get(k).filter(|v| !v.is_null()) else { return Ok(None) };
    let a = v.as_array().filter(|a| a.len() == 2).ok_or_else(|| bad(CMD, format!("`{k}` must be [x, y]")))?;
    let c = |i: usize| a.get(i).and_then(Value::as_f64).filter(|f| f.is_finite() && f.abs() <= 1e6);
    match (c(0), c(1)) {
        (Some(x), Some(y)) => Ok(Some(Vec2::new(x, y))),
        _ => Err(bad(CMD, format!("`{k}` must be two finite numbers"))),
    }
}

fn scale_p(p: &Value, k: &str) -> Result<Option<f64>> {
    match p.get(k).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => v
            .as_f64()
            .filter(|f| f.is_finite() && (0.0..=10000.0).contains(f))
            .map(Some)
            .ok_or_else(|| bad(CMD, format!("`{k}` must be a scale in % (0–10000)"))),
    }
}

fn push_in(s: &mut Session, p: &Value) -> Result<Value> {
    let to = scale_p(p, "to")?.unwrap_or(110.0);
    let from = scale_p(p, "from")?;
    let ease_name = str_p(p, "ease").unwrap_or("bezier");
    let ease =
        interp(ease_name).ok_or_else(|| bad(CMD, format!("unknown ease `{ease_name}` (bezier, easeIn, easeOut, linear, autoBezier, continuousBezier)")))?;
    let angle = match p.get("shutterAngle").filter(|v| !v.is_null()) {
        None => 120.0,
        Some(v) => {
            v.as_f64().filter(|f| f.is_finite() && (0.0..=360.0).contains(f)).ok_or_else(|| bad(CMD, "`shutterAngle` must be 0–360 (0: no motion blur)"))?
        }
    };
    let position = point_p(p, "position")?;
    let from_position = point_p(p, "fromPosition")?;
    let anchor = point_p(p, "anchor")?;
    let move_secs = match f64_p(p, "moveSeconds") {
        None => 0.75,
        Some(v) if v.is_finite() && v > 0.0 && v <= 3600.0 => v,
        Some(_) => return Err(bad(CMD, "`moveSeconds` must be a positive duration")),
    };
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (fw, fh, rate) = (seq.settings.width, seq.settings.height, seq.settings.frame_rate);
    let snap = |t: Tick| rate.tick_of(rate.frame_at(t));

    // the clip that gets the Transform: the one named, or a new adjustment layer over the range
    // (on `track`; checked here, made only once every parameter has been checked)
    let (existing, cs, ce, track) = match clip_p(p, "clip") {
        Some(c) => {
            let (_, it) = seq.find_item(c).ok_or_else(|| bad(CMD, "no such clip in the active sequence"))?;
            (Some(c), it.start, it.end(), None)
        }
        None => {
            let start = time_p(s, p, "start")
                .map(snap)
                .ok_or_else(|| bad(CMD, "need `clip`, or the adjustment layer's range (`startSeconds`/`startFrame`/… and `endSeconds`/`endFrame`/…)"))?;
            let end =
                time_p(s, p, "end").map(snap).ok_or_else(|| bad(CMD, "need the adjustment layer's end (`endSeconds`/`endFrame`/`endTime`/`endTimecode`)"))?;
            if start < Tick::ZERO || end <= start {
                return Err(bad(CMD, "the range must start at 0 or later and end after it starts"));
            }
            let busy = |x: &filmcraft_project::Track| x.items.iter().any(|i| i.start < end && i.end() > start);
            let track = match track_p(s, p, "track", CMD)? {
                Some(t) => {
                    let tr = seq.video_tracks.iter().find(|x| x.id == t).ok_or_else(|| bad(CMD, "`track` must be a video track"))?;
                    if busy(tr) {
                        return Err(bad(CMD, "that track has clips in the range: pick a free video track (or add one with sequence.addTracks)"));
                    }
                    t
                }
                // the lowest free video track above V1 (below any titles already placed higher)
                None => seq
                    .video_tracks
                    .iter()
                    .skip(1)
                    .find(|x| !busy(x))
                    .map(|x| x.id)
                    .ok_or_else(|| bad(CMD, "no free video track above V1 over that range: add one with sequence.addTracks, or pass `track`"))?,
            };
            (None, start, end, Some(track))
        }
    };
    // where the move happens (sequence time), inside the clip
    let m0 = time_p(s, p, "moveStart").or_else(|| if existing.is_some() { time_p(s, p, "start") } else { None }).map(snap).unwrap_or(cs);
    let m1 = time_p(s, p, "moveEnd").map(snap).unwrap_or_else(|| snap(m0 + Tick::from_seconds_f64(move_secs)));
    if m0 < cs || m1 > ce || m1 <= m0 {
        return Err(bad(CMD, "the move must end after it starts, inside the clip"));
    }
    let def = find_effect("transform").ok_or_else(|| bad(CMD, "no Transform effect"))?;
    let created = existing.is_none();
    let clip = match (existing, track) {
        (Some(c), _) => c,
        (None, Some(track)) => {
            let dur = ce - cs;
            let item = s.edit("New Adjustment Layer", |pr, st| {
                let id = pr.add_item("Adjustment Layer", Label::Lavender, ItemKind::AdjustmentLayer { width: fw, height: fh, rate, duration: dur }, None);
                st.project_selection = vec![id];
                Ok(id)
            })?;
            let ids = place_item(s, item, TimeRange::new(Tick::ZERO, dur), cs, Some(track), None, false, "Push-In", None)?;
            ids.first().copied().ok_or_else(|| bad(CMD, "the adjustment layer could not be placed"))?
        }
        (None, None) => return Err(bad(CMD, "no track for the adjustment layer")),
    };
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, it) = seq.find_item(clip).ok_or_else(|| bad(CMD, "the clip is gone"))?;
    // the picture the Transform sees: the sequence frame for an adjustment layer, else the source
    let src = s.project.source_size(it.item).unwrap_or((fw, fh));

    s.edit_sequence("Push-In", |q, _, _| {
        let (_, it) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
        let (t0, t1) = (it.source_time_at(m0), it.source_time_at(m1));
        let index = match it.effects.iter().position(|e| e.effect == "transform") {
            Some(i) => i,
            None => {
                let mut inst = def.instance();
                resolve_auto_points(&mut inst, src, src);
                let pos = it.effects.iter().position(|e| e.def().is_some_and(|d| d.intrinsic)).unwrap_or(it.effects.len());
                it.effects.insert(pos, inst);
                pos
            }
        };
        let e = it.effects.get_mut(index).ok_or_else(|| bad(CMD, "no Transform"))?;
        e.enabled = true;
        let set = |e: &mut EffectInstance, k: &str, v: ParamValue| {
            if let Some(prm) = e.params.get_mut(k) {
                prm.value = v;
            } else {
                e.params.insert(k.to_string(), filmcraft_project::Param::new(v));
            }
        };
        set(e, "uniform_scale", ParamValue::Bool(true));
        set(e, "use_comp_shutter", ParamValue::Bool(false));
        set(e, "shutter_angle", ParamValue::Float(angle));
        let centre = Vec2::new(src.0 as f64 / 2.0, src.1 as f64 / 2.0);
        let known = |v: Vec2| if v.x.is_finite() && v.y.is_finite() { v } else { centre };
        if let Some(a) = anchor {
            // anchor and position move together, so setting the zoom's centre does not shift the
            // picture (the skill's "set both")
            set(e, "anchor", ParamValue::Vec2(a));
            if position.is_none() && e.params.get("position").is_some_and(|x| !x.is_animated()) {
                set(e, "position", ParamValue::Vec2(a));
            }
        }
        let mut keys = Vec::new();
        let mut key = |e: &mut EffectInstance, k: &str, a: ParamValue, b: ParamValue| -> Result<()> {
            let prm = e.params.get_mut(k).ok_or_else(|| bad(CMD, format!("Transform has no `{k}`")))?;
            prm.put_keyframe(t0, a.clone());
            prm.put_keyframe(t1, b.clone());
            for kf in prm.keyframes.iter_mut().filter(|kf| kf.time == t0 || kf.time == t1) {
                kf.interp = ease;
            }
            keys.push(json!({"param": k, "mediaTime": [t0.0, t1.0], "values": [param_json(&a), param_json(&b)]}));
            Ok(())
        };
        let sc0 = from.unwrap_or_else(|| e.params.get("scale_height").map(|x| x.f64_at(t0)).unwrap_or(100.0));
        key(e, "scale_height", ParamValue::Float(sc0), ParamValue::Float(to))?;
        if let Some(p1) = position {
            let p0 = from_position.unwrap_or_else(|| known(e.params.get("position").map(|x| x.vec2_at(t0)).unwrap_or(centre)));
            key(e, "position", ParamValue::Vec2(p0), ParamValue::Vec2(p1))?;
        }
        Ok(json!({"clip": clip.0, "effect": index, "created": created, "keys": keys}))
    })
}

fn param_json(v: &ParamValue) -> Value {
    match v {
        ParamValue::Float(f) => json!(f),
        ParamValue::Vec2(p) => json!([p.x, p.y]),
        _ => Value::Null,
    }
}
