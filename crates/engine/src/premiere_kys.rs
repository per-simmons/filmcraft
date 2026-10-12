//! Premiere Pro keyboard shortcut files (`.kys`), read by `shortcuts.import`.
//!
//! A `.kys` file is the XML Premiere writes for a keyboard shortcut preset (its user profile's
//! `Mac/` or `Win/` folder holds one per preset):
//!
//! ```xml
//! <PremiereData Version="3"><shortcuts Version="5">
//!   <platform>macintosh</platform>
//!   <context.global Version="1">
//!     <itemcount>2</itemcount>
//!     <item.0 Version="1">
//!       <virtualkey>2147483723</virtualkey>
//!       <modifier.command>true</modifier.command><modifier.opt>false</modifier.opt>
//!       <modifier.shift>false</modifier.shift><modifier.ctrl>false</modifier.ctrl>
//!       <commandname>cmd.sequence.razorateditline</commandname>
//!     </item.0>
//!     <item.1 Version="1"><commandname>cmd.file.capture</commandname></item.1>
//!   </context.global>
//!   <context.timeline Version="1">…</context.timeline>
//! </shortcuts></PremiereData>
//! ```
//!
//! Each `context.*` element is a scope (`global` = application-wide, the others are panels); each
//! item names one command and, when it has a key, its `virtualkey` and modifiers. An item without a
//! key is a command the preset leaves unassigned. Key codes, worked out from the files themselves:
//! with the top bit set (`0x8000_0000 | c`) the key is the one that types the ASCII character `c`
//! (`'K'`, `'/'`, `'['`…); below that are the named keys (1 Space, 2 Backspace, 3 Tab, 4 Return,
//! 7–18 F1–F12, 35 forward Delete, 36 Home, 37 End, 38 Page Up, 39 Page Down, 42–45 ← → ↑ ↓); 0 is
//! "no key". Only the key facts of the file are used: command names map onto FilmCraft's own
//! command ids through [`COMMANDS`] (FilmCraft mirrors Premiere's command names, so the table is
//! built from FilmCraft's labels and menu paths).

use std::collections::BTreeSet;

use serde::Serialize;

use crate::shortcuts::{Binding, Chord, Mods, Platform, canon_key};

/// Files larger than this are refused (Premiere's own are a few hundred KB).
const MAX_BYTES: u64 = 32 << 20;
/// At most this many items are read (Premiere has ~1,500 commands).
const MAX_ITEMS: usize = 100_000;

/// One item of a `.kys` file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KysItem {
    /// The context without its `context.` prefix (`global`, `timeline`, `project`…).
    pub context: String,
    /// Premiere's command name (`cmd.sequence.razorateditline`, `uif.window.Timelines`).
    pub command: String,
    /// The key; None = the command has no key in this preset.
    pub chord: Option<Chord>,
    /// A key code this reader doesn't know (reported, not imported).
    pub unknown_key: Option<u64>,
}

/// A parsed `.kys` file.
#[derive(Clone, Debug)]
pub struct KysFile {
    pub platform: Platform,
    pub items: Vec<KysItem>,
}

/// Does `bytes` look like a Premiere keyboard shortcuts file?
pub fn is_kys(bytes: &[u8]) -> bool {
    let head = bytes.get(..bytes.len().min(1024)).unwrap_or_default();
    String::from_utf8_lossy(head).contains("<PremiereData")
}

/// Read a `.kys` file.
pub fn read(path: &std::path::Path) -> Result<KysFile, String> {
    let len = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?.len();
    if len > MAX_BYTES {
        return Err(format!("{}: {len} bytes is too large for a keyboard shortcuts file", path.display()));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&String::from_utf8_lossy(&bytes)).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse the text of a `.kys` file.
pub fn parse(text: &str) -> Result<KysFile, String> {
    let opts = roxmltree::ParsingOptions { allow_dtd: false, nodes_limit: 4_000_000, ..Default::default() };
    let doc = roxmltree::Document::parse_with_options(text, opts).map_err(|e| format!("not a Premiere keyboard shortcuts file: {e}"))?;
    let root = doc.root_element();
    if root.tag_name().name() != "PremiereData" {
        return Err("not a Premiere keyboard shortcuts file (no <PremiereData>)".into());
    }
    let shortcuts = root.children().find(|n| n.is_element() && n.tag_name().name() == "shortcuts").ok_or("no <shortcuts> in the file")?;
    let platform = match child_text(shortcuts, "platform").map(str::to_ascii_lowercase).as_deref() {
        Some(p) if p.starts_with("win") => Platform::Windows,
        _ => Platform::Mac,
    };
    let mut items = Vec::new();
    for ctx in shortcuts.children().filter(|n| n.is_element()) {
        let Some(context) = ctx.tag_name().name().strip_prefix("context.") else { continue };
        for it in ctx.children().filter(|n| n.is_element() && n.tag_name().name().starts_with("item.")) {
            if items.len() >= MAX_ITEMS {
                return Err(format!("more than {MAX_ITEMS} shortcuts"));
            }
            let Some(command) = child_text(it, "commandname").map(str::trim).filter(|c| !c.is_empty()) else { continue };
            let flag = |n: &str| child_text(it, n).is_some_and(|v| v.trim().eq_ignore_ascii_case("true"));
            let vk = child_text(it, "virtualkey").and_then(|v| v.trim().parse::<u64>().ok()).filter(|v| *v != 0);
            let (mut chord, mut unknown_key) = (None, None);
            if let Some(vk) = vk {
                match decode_key(vk) {
                    Some(key) => {
                        let (cmd, ctrl) = if platform.is_mac() {
                            (flag("modifier.command"), flag("modifier.ctrl"))
                        } else {
                            // Windows: Ctrl is the primary modifier (FilmCraft's `Cmd`)
                            (flag("modifier.ctrl") || flag("modifier.command"), false)
                        };
                        chord = Some(Chord { mods: Mods { cmd, ctrl, alt: flag("modifier.opt") || flag("modifier.alt"), shift: flag("modifier.shift") }, key });
                    }
                    None => unknown_key = Some(vk),
                }
            }
            items.push(KysItem { context: context.to_string(), command: command.to_string(), chord, unknown_key });
        }
    }
    Ok(KysFile { platform, items })
}

fn child_text<'a>(n: roxmltree::Node<'a, '_>, name: &str) -> Option<&'a str> {
    n.children().find(|c| c.is_element() && c.tag_name().name() == name).and_then(|c| c.text())
}

/// The canonical key of a Premiere virtual key code.
pub fn decode_key(vk: u64) -> Option<&'static str> {
    const CHAR_BIT: u64 = 0x8000_0000;
    if vk & CHAR_BIT != 0 {
        let c = char::from_u32(u32::try_from(vk & !CHAR_BIT).ok()?)?;
        if !c.is_ascii_graphic() {
            return None;
        }
        return canon_key(&c.to_ascii_uppercase().to_string());
    }
    Some(match vk {
        1 => "Space",
        2 => "Backspace",
        3 => "Tab",
        4 => "Enter",
        7..=18 => *["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"].get(usize::try_from(vk - 7).ok()?)?,
        35 => "Delete",
        36 => "Home",
        37 => "End",
        38 => "PageUp",
        39 => "PageDown",
        42 => "Left",
        43 => "Right",
        44 => "Up",
        45 => "Down",
        _ => return None,
    })
}

/// FilmCraft's scope for a Premiere context: `Some(None)` = application-wide, `Some(Some(panel))`,
/// `None` = a Premiere panel FilmCraft doesn't have.
pub fn panel_of(context: &str) -> Option<Option<&'static str>> {
    Some(Some(match context {
        "global" => return Some(None),
        "timeline" => "Timeline",
        "program.monitor" => "Program",
        "source.monitor" => "Source",
        "project" => "Project",
        "effectcontrols" => "Effect Controls",
        "effects" => "Effects",
        "history" => "History",
        "markerList" => "Markers",
        "mediabrowser" => "Media Browser",
        "metadata_editor" => "Metadata",
        "audiomixer" => "Audio Track Mixer",
        "audioclipmixer" => "Audio Clip Mixer",
        "Graphics" => "Essential Graphics",
        "com.adobe.dva.text" => "Text",
        "Properties2" => "Properties",
        "color" => "Lumetri Color",
        _ => return None,
    }))
}

/// Premiere command name → FilmCraft command id, where one context gives the command another
/// meaning (`context`, Premiere command, FilmCraft command).
pub const IN_CONTEXT: &[(&str, &str, &str)] = &[
    ("project", "cmd.edit.clear", "project.delete"),
    ("project", "cmd.project.deletewithoptions", "project.delete"),
    ("history", "cmd.history.step.backward", "edit.undo"),
    ("history", "cmd.history.step.forward", "edit.redo"),
];

/// Premiere command name → FilmCraft command id.
pub const COMMANDS: &[(&str, &str)] = &[
    // Tools
    ("cmd.tools.01pointer", "tool.selection"),
    ("cmd.tools.02trackselectforward", "tool.trackSelectForward"),
    ("cmd.tools.02_5trackselectbackward", "tool.trackSelectBackward"),
    ("cmd.tools.03ripple", "tool.ripple"),
    ("cmd.tools.04roll", "tool.rolling"),
    ("cmd.tools.05ratestretch", "tool.rateStretch"),
    ("cmd.tools.06razor", "tool.razor"),
    ("cmd.tools.07slip", "tool.slip"),
    ("cmd.tools.08slide", "tool.slide"),
    ("cmd.tools.09pen", "tool.pen"),
    ("cmd.tools.10hand", "tool.hand"),
    ("cmd.tools.11zoom", "tool.zoom"),
    ("cmd.tools.12text", "tool.type"),
    ("cmd.tools.14verticaltype", "tool.verticalType"),
    ("cmd.tools.16Remix", "tool.remix"),
    // File
    ("cmd.file.new.project", "file.newProject"),
    ("cmd.file.new.sequence", "file.newSequence"),
    ("cmd.file.new.bin", "file.newBin"),
    ("cmd.file.newbinfromselection", "file.newBinFromSelection"),
    ("cmd.file.new.smartbin", "file.newSearchBin"),
    ("cmd.file.new.offlinefile", "file.newOfflineFile"),
    ("cmd.file.new.adjustmentlayer", "file.newAdjustmentLayer"),
    ("cmd.file.openproject", "file.open"),
    ("cmd.file.closepanel", "file.close"),
    ("cmd.file.close", "file.closeProject"),
    ("cmd.file.closeall", "file.closeAllProjects"),
    ("cmd.file.closeallother", "file.closeAllOtherProjects"),
    ("cmd.file.save", "file.save"),
    ("cmd.file.saveas", "file.saveAs"),
    ("cmd.file.savecopy", "file.saveCopy"),
    ("cmd.file.saveall", "file.saveAll"),
    ("cmd.file.saveastemplate", "file.saveAsTemplate"),
    ("cmd.file.revert", "file.revert"),
    ("cmd.file.import", "file.import"),
    ("cmd.file.importfrombrowser", "file.importFromMediaBrowser"),
    ("cmd.file.export.movie", "mode.export"),
    ("cmd.file.export.sendtoqueue", "export.queue.add"),
    ("cmd.export.frame", "file.exportFrame"),
    ("cmd.file.properties.selection", "file.mediaProperties"),
    ("cmd.file.exit", "app.quit"),
    ("cmd.window.mode.Import", "mode.import"),
    ("cmd.window.mode.Edit", "mode.edit"),
    ("cmd.window.mode.Export", "mode.export"),
    // Edit
    ("cmd.edit.undo", "edit.undo"),
    ("cmd.edit.redo", "edit.redo"),
    ("cmd.edit.cut", "edit.cut"),
    ("cmd.edit.copy", "edit.copy"),
    ("cmd.edit.paste", "edit.paste"),
    ("cmd.edit.pasteinsert", "edit.pasteInsert"),
    ("cmd.edit.pasteattributes", "edit.pasteAttributes"),
    ("cmd.edit.clear", "edit.clear"),
    ("cmd.graphics.clear", "edit.clear"),
    ("cmd.edit.rippledelete", "edit.rippleDelete"),
    ("cmd.timeline.ripple.delete", "edit.rippleDelete"),
    ("cmd.edit.duplicate", "edit.duplicate"),
    ("cmd.edit.selectall", "edit.selectAll"),
    ("cmd.edit.selectallmatching", "edit.selectAllMatching"),
    ("cmd.edit.deselectall", "edit.deselectAll"),
    ("cmd.edit.find", "edit.find"),
    ("cmd.edit.findnext", "edit.findNext"),
    ("cmd.edit.labelgroup", "edit.selectLabelGroup"),
    ("cmd.edit.editoriginal", "edit.editOriginal"),
    ("cmd.edit.keyboardshortcuts", "app.keyboardShortcuts"),
    ("cmd.edit.preferences.general", "app.settings.general"),
    ("cmd.edit.preferences.audio", "app.settings.audio"),
    ("cmd.edit.preferences.audiohardware", "app.settings.audioHardware"),
    ("cmd.edit.preferences.autosaveundo", "app.settings.autoSave"),
    ("cmd.edit.preferences.color", "app.settings.color"),
    ("cmd.edit.preferences.labelcolors", "app.settings.labels"),
    ("cmd.edit.preferences.media", "app.settings.media"),
    ("cmd.edit.preferences.mediaanalysis", "app.settings.mediaAnalysis"),
    ("cmd.edit.preferences.mediacache", "app.settings.mediaCache"),
    ("cmd.edit.preferences.memory", "app.settings.memory"),
    ("cmd.edit.preferences.playback", "app.settings.playback"),
    ("cmd.edit.preferences.timeline", "app.settings.timeline"),
    ("cmd.edit.preferences.trim", "app.settings.trim"),
    ("cmd.edit.preferences.userinterface", "app.settings.appearance"),
    ("cmd.project.removeunused", "edit.removeUnused"),
    ("cmd.project.consolidateduplicates", "edit.consolidateDuplicates"),
    // Clip
    ("cmd.clip.makesubclip", "clip.makeSubclip"),
    ("cmd.clip.editsubclip", "clip.editSubclip"),
    ("cmd.clip.editoffline", "clip.editOffline"),
    ("cmd.clip.audiooptions.sourcechannelmappings", "clip.audioChannels"),
    ("cmd.clip.audiooptions.gain", "clip.audioGain"),
    ("cmd.clip.audiooptions.breakouttomono", "clip.breakoutToMono"),
    ("cmd.clip.extractaudio", "clip.extractAudio"),
    ("cmd.clip.audiooptions.nudgevolumeup", "clip.nudgeVolumeUp1"),
    ("cmd.clip.audiooptions.nudgevolumeup3", "clip.nudgeVolumeUp3"),
    ("cmd.clip.audiooptions.nudgevolumedown", "clip.nudgeVolumeDown1"),
    ("cmd.clip.audiooptions.nudgevolumedown3", "clip.nudgeVolumeDown3"),
    ("cmd.clip.videooptions.frameholdoptions", "clip.frameHold"),
    ("cmd.clip.videooptions.insertframeholdsegment", "clip.insertFrameHoldSegment"),
    ("cmd.clip.framesample", "clip.timeInterpolation.frameSampling"),
    ("cmd.clip.frameblend", "clip.timeInterpolation.frameBlending"),
    ("cmd.clip.opticalflow", "clip.timeInterpolation.opticalFlow"),
    ("cmd.clip.scaletoframesize", "clip.scaleToFrameSize"),
    // Set to Frame Size scales the picture so it fits the frame, as Fit to Frame does
    ("cmd.clip.settoframesize", "clip.fitToFrame"),
    ("cmd.clip.fittoframe", "clip.fitToFrame"),
    ("cmd.clip.fillframe", "clip.fillFrame"),
    ("cmd.clip.speed", "clip.speedDuration"),
    ("cmd.clip.enable", "clip.enable"),
    ("cmd.clip.linkaudioandvideo", "clip.link"),
    ("cmd.clip.group", "clip.group"),
    ("cmd.clip.ungroup", "clip.ungroup"),
    ("cmd.clip.insert", "source.insert"),
    ("cmd.clip.overlay", "source.overwrite"),
    ("cmd.clip.replaceclip.fromsourcemonitor", "clip.replaceFromSource"),
    ("cmd.clip.replaceclip.fromsourcemonitor.matchframe", "clip.replaceFromSourceMatchFrame"),
    ("cmd.clip.replaceclip.frombin", "clip.replaceFromBin"),
    ("cmd.clip.restorecaptions", "clip.restoreCaptionsFromSource"),
    ("cmd.clip.updatemetadata", "clip.updateMetadata"),
    ("cmd.clip.multicam.toggle", "clip.multicamEnable"),
    ("cmd.clip.multicam.flatten", "clip.multicamFlatten"),
    ("cmd.clip.remix.enable", "clip.remix.enable"),
    ("cmd.clip.remix.properties", "clip.remix.properties"),
    ("cmd.clip.remix.revert", "clip.remix.revert"),
    ("cmd.posterframe.set", "clip.setPosterFrame"),
    ("cmd.posterframe.clear", "clip.clearPosterFrame"),
    ("cmd.sequence.increaseclipvolume", "clip.volumeUp"),
    ("cmd.sequence.decreaseclipvolume", "clip.volumeDown"),
    ("cmd.sequence.increaseclipvolumemany", "clip.volumeUpMany"),
    ("cmd.sequence.decreaseclipvolumemany", "clip.volumeDownMany"),
    ("cmd.edit.revealinproject", "clip.revealInProject"),
    // Sequence
    ("cmd.sequence.sequencesettingsgeneral", "sequence.settings"),
    ("cmd.sequence.previewyellow", "sequence.renderEffectsInToOut"),
    ("cmd.sequence.preview", "sequence.renderInToOut"),
    ("cmd.sequence.previewselection", "sequence.renderSelection"),
    ("cmd.sequence.previewaudio", "sequence.renderAudio"),
    ("cmd.sequence.deletevideopreviews", "sequence.deleteRenderFiles"),
    ("cmd.sequence.deleteworkareavideopreviews", "sequence.deleteRenderFilesInToOut"),
    ("cmd.sequence.matchframe", "sequence.matchFrame"),
    ("cmd.sequence.reversematchframe", "sequence.reverseMatchFrame"),
    ("cmd.sequence.razorateditline", "sequence.addEdit"),
    ("cmd.sequence.razorateditline.all", "sequence.addEditAllTracks"),
    ("cmd.sequence.applydefaultvideotransition", "sequence.applyVideoTransition"),
    ("cmd.sequence.applydefaultaudiotransition", "sequence.applyAudioTransition"),
    ("cmd.sequence.applydefaulttransitions", "trim.applyDefaultTransition"),
    ("cmd.sequence.lift", "sequence.lift"),
    ("cmd.sequence.extract", "sequence.extract"),
    ("cmd.sequence.findnextsequencegap", "sequence.goToNextGap"),
    ("cmd.sequence.findprevioussequencegap", "sequence.goToPrevGap"),
    ("cmd.sequence.findnexttrackgap", "sequence.goToNextGapInTrack"),
    ("cmd.sequence.findprevioustrackgap", "sequence.goToPrevGapInTrack"),
    ("cmd.sequence.snap", "sequence.snap"),
    ("cmd.sequence.linkedselection", "sequence.linkedSelection"),
    ("cmd.sequence.selectionfollowsplayhead", "sequence.selectionFollowsPlayhead"),
    ("cmd.sequence.showthroughedits", "sequence.showThroughEdits"),
    ("cmd.sequence.jointhroughedits", "sequence.joinThroughEdits"),
    ("cmd.sequence.makesubsequence", "sequence.makeSubsequence"),
    ("cmd.sequence.transcribeasset", "sequence.transcribe"),
    ("cmd.sequence.addtracks", "sequence.addTracks"),
    ("cmd.sequence.deletetracks", "sequence.deleteTracks"),
    ("cmd.sequence.nestsourcesequence", "sequence.nestSequences"),
    ("cmd.sequence.normalizetrack", "sequence.normalizeMixTrack"),
    ("cmd.timeline.sequence.revealinproject", "sequence.revealInProject"),
    ("cmd.sequence.newsequencefromselection", "file.newSequenceFromClip"),
    ("cmd.tlnav.reveal.nested.sequence", "sequence.revealNested"),
    // Trimming
    ("cmd.sequence.extendselectededittoplayhead", "trim.extendToPlayhead"),
    ("cmd.sequence.extendpreviousedittoplayhead", "trim.extendPreviousEdit"),
    ("cmd.sequence.extendnextedittoplayhead", "trim.extendNextEdit"),
    ("cmd.sequence.rippletrimpreviousedittoplayhead", "trim.ripplePrevious"),
    ("cmd.sequence.rippletrimnextedittoplayhead", "trim.rippleNext"),
    ("cmd.tlnav.trim.in.to.cti", "trim.previous"),
    ("cmd.tlnav.trim.out.to.cti", "trim.next"),
    ("cmd.sequence.toggletrimtype", "trim.toggleType"),
    ("cmd.sequence.trimbackward", "trim.backward"),
    ("cmd.sequence.trimforward", "trim.forward"),
    ("cmd.sequence.trimbackwardmany", "trim.backwardMany"),
    ("cmd.sequence.trimforwardmany", "trim.forwardMany"),
    ("cmd.tlnav.select.nearest.edit.as.ripple.in", "trim.selectNearestRippleIn"),
    ("cmd.tlnav.select.nearest.edit.as.ripple.out", "trim.selectNearestRippleOut"),
    ("cmd.tlnav.select.nearest.edit.as.roll", "trim.selectNearestRoll"),
    ("cmd.tlnav.select.nearest.edit.as.trim.in", "trim.selectNearestTrimIn"),
    ("cmd.tlnav.select.nearest.edit.as.trim.out", "trim.selectNearestTrimOut"),
    // Markers, In and Out
    ("cmd.common.setin", "markers.markIn"),
    ("cmd.common.setout", "markers.markOut"),
    ("cmd.marker.setsequenceinoutmarkeraroundtargetclip", "markers.markClip"),
    ("cmd.marker.setsequenceinoutmarkeraroundselection.out", "markers.markSelection"),
    ("cmd.goto.in", "markers.goToIn"),
    ("cmd.goto.out", "markers.goToOut"),
    ("cmd.clear.in", "markers.clearIn"),
    ("cmd.clear.out", "markers.clearOut"),
    ("cmd.clear.inandout", "markers.clearInOut"),
    ("cmd.set.marker", "markers.add"),
    ("cmd.set.rangedmarker", "markers.addRange"),
    ("cmd.set.rangedmarker.inandout", "markers.addRangeInOut"),
    ("cmd.marker.gotomarker.next", "markers.goNext"),
    ("cmd.marker.gotomarker.previous", "markers.goPrev"),
    ("cmd.marker.clearmarker.current", "markers.clearCurrent"),
    ("cmd.marker.clearmarker.all", "markers.clearAll"),
    ("cmd.marker.setchaptermarker", "markers.addChapter"),
    ("cmd.marker.setflashcuemarker", "markers.addFlashCue"),
    ("cmd.marker.style.ripplesequencemarkers", "markers.rippleSequenceMarkers"),
    ("cmd.marker.copypaste.includessequencemarkers", "markers.copyPasteIncludesSequenceMarkers"),
    // Navigation and playback
    ("cmd.transport.toggleplay", "playback.toggle"),
    ("cmd.transport.shuttle.right", "playback.forward"),
    ("cmd.transport.shuttle.stop", "playback.stop"),
    ("cmd.transport.shuttle.left", "playback.reverse"),
    ("cmd.transport.shuttle.slow.right", "playback.slowForward"),
    ("cmd.transport.shuttle.slow.left", "playback.slowReverse"),
    ("cmd.transport.playedit", "playback.playAround"),
    ("cmd.transport.playintoout", "playback.inToOut"),
    ("cmd.transport.play.fat", "playback.inToOutPreroll"),
    ("cmd.transport.play.ctitoout", "playback.toOut"),
    ("cmd.monitor.loop", "playback.loop"),
    ("cmd.transport.step.forward", "playhead.stepForward"),
    ("cmd.transport.step.back", "playhead.stepBack"),
    ("cmd.transport.step.forward.five", "playhead.stepForward5"),
    ("cmd.transport.step.back.five", "playhead.stepBack5"),
    ("cmd.transport.sequence.start", "playhead.start"),
    ("cmd.transport.sequence.end", "playhead.end"),
    ("cmd.transport.selectedclip.start", "playhead.selectedClipStart"),
    ("cmd.transport.selectedclip.end", "playhead.selectedClipEnd"),
    ("cmd.tlnav.next.edit", "playhead.nextEdit"),
    ("cmd.tlnav.prev.edit", "playhead.prevEdit"),
    ("cmd.tlnav.next.edit.any.track", "playhead.nextEditAnyTrack"),
    ("cmd.tlnav.prev.edit.any.track", "playhead.prevEditAnyTrack"),
    ("cmd.toggle.audio.scrubbing", "audio.toggleScrubbing"),
    // Timeline
    ("cmd.tlnav.select.clip.at.playhead", "timeline.selectClipAtPlayhead"),
    ("cmd.tlnav.select.next.clip", "timeline.selectNextClip"),
    ("cmd.tlnav.select.previous.clip", "timeline.selectPrevClip"),
    ("cmd.timeline.nudge.left.one", "timeline.nudgeLeft"),
    ("cmd.timeline.nudge.right.one", "timeline.nudgeRight"),
    ("cmd.timeline.nudge.left.several", "timeline.nudgeLeft5"),
    ("cmd.timeline.nudge.right.several", "timeline.nudgeRight5"),
    ("cmd.timeline.nudge.up", "timeline.nudgeUp"),
    ("cmd.timeline.nudge.down", "timeline.nudgeDown"),
    ("cmd.timeline.slip.left.one", "timeline.slipLeft"),
    ("cmd.timeline.slip.right.one", "timeline.slipRight"),
    ("cmd.timeline.slip.left.several", "timeline.slipLeft5"),
    ("cmd.timeline.slip.right.several", "timeline.slipRight5"),
    ("cmd.timeline.slide.left.one", "timeline.slideLeft"),
    ("cmd.timeline.slide.right.one", "timeline.slideRight"),
    ("cmd.timeline.slide.left.several", "timeline.slideLeft5"),
    ("cmd.timeline.slide.right.several", "timeline.slideRight5"),
    ("cmd.timeline.paste.to.same.track", "edit.paste"),
    ("cmd.timeline.pasteinsert.to.same.track", "edit.pasteInsert"),
    ("cmd.timeline.increase.video.tracks.height", "timeline.increaseVideoHeight"),
    ("cmd.timeline.decrease.video.tracks.height", "timeline.decreaseVideoHeight"),
    ("cmd.timeline.increase.audio.tracks.height", "timeline.increaseAudioHeight"),
    ("cmd.timeline.decrease.audio.tracks.height", "timeline.decreaseAudioHeight"),
    ("cmd.timeline.expand.all.tracks", "timeline.expandAllTracks"),
    ("cmd.timeline.minimize.all.tracks", "timeline.minimizeAllTracks"),
    ("cmd.timeline.show.next.screen", "timeline.nextScreen"),
    ("cmd.timeline.show.previous.screen", "timeline.prevScreen"),
    ("cmd.timeline.goto.next.caption.trackitem", "captions.next"),
    ("cmd.timeline.goto.prev.caption.trackitem", "captions.previous"),
    ("cmd.timeline.show.all.caption.tracks", "captions.showAll"),
    ("cmd.timeline.show.active.caption.track.only", "captions.showActiveOnly"),
    ("cmd.timeline.hide.all.caption.tracks", "captions.hideAll"),
    ("cmd.tlnav.toggle.all.target.video", "timeline.toggleAllVideoTargets"),
    ("cmd.tlnav.toggle.all.target.audio", "timeline.toggleAllAudioTargets"),
    ("cmd.tlnav.toggle.all.source.video", "timeline.toggleAllSourceVideo"),
    ("cmd.tlnav.toggle.all.source.audio", "timeline.toggleAllSourceAudio"),
    ("cmd.tlnav.move.all.target.video.up", "timeline.moveVideoTargetsUp"),
    ("cmd.tlnav.move.all.target.video.down", "timeline.moveVideoTargetsDown"),
    ("cmd.tlnav.move.all.target.audio.up", "timeline.moveAudioTargetsUp"),
    ("cmd.tlnav.move.all.target.audio.down", "timeline.moveAudioTargetsDown"),
    ("cmd.sequence.toggleaudiotrackmutes", "timeline.toggleMuteTargetedAudio"),
    ("cmd.sequence.toggleaudiotracksolos", "timeline.toggleSoloTargetedAudio"),
    ("cmd.sequence.togglevideotrackoutputs", "timeline.toggleOutputTargetedVideo"),
    ("cmd.tlnav.toggle.target.video.1", "timeline.toggleTargetV1"),
    ("cmd.tlnav.toggle.target.video.2", "timeline.toggleTargetV2"),
    ("cmd.tlnav.toggle.target.video.3", "timeline.toggleTargetV3"),
    ("cmd.tlnav.toggle.target.video.4", "timeline.toggleTargetV4"),
    ("cmd.tlnav.toggle.target.video.5", "timeline.toggleTargetV5"),
    ("cmd.tlnav.toggle.target.video.6", "timeline.toggleTargetV6"),
    ("cmd.tlnav.toggle.target.video.7", "timeline.toggleTargetV7"),
    ("cmd.tlnav.toggle.target.video.8", "timeline.toggleTargetV8"),
    ("cmd.tlnav.toggle.target.audio.1", "timeline.toggleTargetA1"),
    ("cmd.tlnav.toggle.target.audio.2", "timeline.toggleTargetA2"),
    ("cmd.tlnav.toggle.target.audio.3", "timeline.toggleTargetA3"),
    ("cmd.tlnav.toggle.target.audio.4", "timeline.toggleTargetA4"),
    ("cmd.tlnav.toggle.target.audio.5", "timeline.toggleTargetA5"),
    ("cmd.tlnav.toggle.target.audio.6", "timeline.toggleTargetA6"),
    ("cmd.tlnav.toggle.target.audio.7", "timeline.toggleTargetA7"),
    ("cmd.tlnav.toggle.target.audio.8", "timeline.toggleTargetA8"),
    // Captions
    ("cmd.caption.add.track", "captions.newTrack"),
    ("cmd.caption.add.trackitem", "captions.add"),
    ("cmd.caption.edit.split", "captions.split"),
    ("cmd.caption.edit.merge", "captions.merge"),
    // Multi-camera
    ("cmd.multicam.toggle.record", "multicam.recordToggle"),
    ("cmd.multicam.toggle.multicam.view", "multicam.toggleView"),
    ("cmd.multicam.audio.follows.video", "multicam.audioFollowsVideo"),
    ("cmd.multicam.page.next", "multicam.nextPage"),
    ("cmd.multicam.page.previous", "multicam.prevPage"),
    ("cmd.multicam.selection.top.down", "multicam.selectionTopDown"),
    ("cmd.multicam.enable.auto.decimation", "multicam.autoAdjustQuality"),
    ("cmd.multicam.transmit.gridview", "multicam.transmitView"),
    ("cmd.multicam.choosenocut.camera1", "multicam.selectCamera1"),
    ("cmd.multicam.choosenocut.camera2", "multicam.selectCamera2"),
    ("cmd.multicam.choosenocut.camera3", "multicam.selectCamera3"),
    ("cmd.multicam.choosenocut.camera4", "multicam.selectCamera4"),
    ("cmd.multicam.choosenocut.camera5", "multicam.selectCamera5"),
    ("cmd.multicam.choosenocut.camera6", "multicam.selectCamera6"),
    ("cmd.multicam.choosenocut.camera7", "multicam.selectCamera7"),
    ("cmd.multicam.choosenocut.camera8", "multicam.selectCamera8"),
    ("cmd.multicam.choosenocut.camera9", "multicam.selectCamera9"),
    ("cmd.multicam.choose.camera.1", "multicam.cutToCamera1"),
    ("cmd.multicam.choose.camera.2", "multicam.cutToCamera2"),
    ("cmd.multicam.choose.camera.3", "multicam.cutToCamera3"),
    ("cmd.multicam.choose.camera.4", "multicam.cutToCamera4"),
    ("cmd.multicam.choose.camera.5", "multicam.cutToCamera5"),
    ("cmd.multicam.choose.camera.6", "multicam.cutToCamera6"),
    ("cmd.multicam.choose.camera.7", "multicam.cutToCamera7"),
    ("cmd.multicam.choose.camera.8", "multicam.cutToCamera8"),
    ("cmd.multicam.choose.camera.9", "multicam.cutToCamera9"),
    // Graphics and titles
    ("cmd.graphics.add.text", "graphics.newText"),
    ("cmd.graphics.add.text.vertical", "graphics.newVerticalText"),
    ("cmd.graphics.add.shape.rectangle", "graphics.newRectangle"),
    ("cmd.graphics.add.shape.ellipse", "graphics.newEllipse"),
    ("cmd.graphics.add.shape.polygon", "graphics.newPolygon"),
    ("cmd.graphics.move.graphic.layer.to.top", "graphics.bringToFront"),
    ("cmd.graphics.move.graphic.layer.up", "graphics.bringForward"),
    ("cmd.graphics.move.graphic.layer.down", "graphics.sendBackward"),
    ("cmd.graphics.move.graphic.layer.to.bottom", "graphics.sendToBack"),
    ("cmd.graphics.select.next.graphic", "graphics.selectNextGraphic"),
    ("cmd.graphics.select.previous.graphic", "graphics.selectPreviousGraphic"),
    ("cmd.graphics.select.next.graphic.layer", "graphics.selectNextLayer"),
    ("cmd.graphics.select.previous.graphic.layer", "graphics.selectPreviousLayer"),
    ("cmd.graphics.reset.duration", "graphics.resetDuration"),
    ("cmd.graphics.upgrade.to.source", "graphics.upgradeToSourceGraphic"),
    ("cmd.graphics.enter.text.edit", "graphics.beginTextEditing"),
    ("cmd.graphics.dialog.replacefont", "file.replaceFonts"),
    ("cmd.graphics.distribute.vertical.center", "graphics.distributeVertically"),
    ("cmd.graphics.distribute.vertical.space", "graphics.distributeSpaceVertically"),
    ("cmd.graphics.distribute.horizontal.center", "graphics.distributeHorizontally"),
    ("cmd.graphics.distribute.horizontal.space", "graphics.distributeSpaceHorizontally"),
    ("cmd.graphicsinspector.alignleft", "graphics.alignTextLeft"),
    ("cmd.graphicsinspector.aligncenter", "graphics.alignTextCenter"),
    ("cmd.graphicsinspector.alignright", "graphics.alignTextRight"),
    ("cmd.titler.misc.size.one.inc", "graphics.fontSizeUp"),
    ("cmd.titler.misc.size.one.dec", "graphics.fontSizeDown"),
    ("cmd.titler.misc.size.five.inc", "graphics.fontSizeUp5"),
    ("cmd.titler.misc.size.five.dec", "graphics.fontSizeDown5"),
    ("cmd.titler.misc.lead.one.inc", "graphics.leadingUp"),
    ("cmd.titler.misc.lead.one.dec", "graphics.leadingDown"),
    ("cmd.titler.misc.lead.five.inc", "graphics.leadingUp5"),
    ("cmd.titler.misc.lead.five.dec", "graphics.leadingDown5"),
    ("cmd.monitor.nudge.left.one", "graphics.nudgeLeft"),
    ("cmd.monitor.nudge.right.one", "graphics.nudgeRight"),
    ("cmd.monitor.nudge.up.one", "graphics.nudgeUp"),
    ("cmd.monitor.nudge.down.one", "graphics.nudgeDown"),
    ("cmd.monitor.nudge.left.five", "graphics.nudgeLeft5"),
    ("cmd.monitor.nudge.right.five", "graphics.nudgeRight5"),
    ("cmd.monitor.nudge.up.five", "graphics.nudgeUp5"),
    ("cmd.monitor.nudge.down.five", "graphics.nudgeDown5"),
    // View and monitors
    ("cmd.zoom.in", "view.zoomIn"),
    ("cmd.zoom.out", "view.zoomOut"),
    ("cmd.tlnav.zoomto.sequence", "view.zoomToSequence"),
    ("cmd.monitor.rulers", "view.showRulers"),
    ("cmd.monitor.guides", "view.showGuides"),
    ("cmd.monitor.lockguides", "view.lockGuides"),
    ("cmd.monitor.addguide", "view.addGuide"),
    ("cmd.monitor.removeguides", "view.clearGuides"),
    ("cmd.monitor.snapping", "view.snapInProgramMonitor"),
    ("cmd.monitor.wingtip.safemargins", "view.safeMargins"),
    ("cmd.monitor.saveguides", "view.guideTemplates.save"),
    ("cmd.monitor.manageguides", "view.guideTemplates.manage"),
    ("cmd.monitor.playback.qualityishigh", "view.highQualityPlayback"),
    ("cmd.monitor.playback.resolution.full", "view.playbackRes.full"),
    ("cmd.monitor.playback.resolution.half", "view.playbackRes.half"),
    ("cmd.monitor.playback.resolution.quarter", "view.playbackRes.quarter"),
    ("cmd.monitor.playback.resolution.eighth", "view.playbackRes.eighth"),
    ("cmd.monitor.playback.resolution.sixteenth", "view.playbackRes.sixteenth"),
    ("cmd.monitor.paused.resolution.full", "view.pausedRes.full"),
    ("cmd.monitor.paused.resolution.half", "view.pausedRes.half"),
    ("cmd.monitor.paused.resolution.quarter", "view.pausedRes.quarter"),
    ("cmd.monitor.paused.resolution.eighth", "view.pausedRes.eighth"),
    ("cmd.monitor.paused.resolution.sixteenth", "view.pausedRes.sixteenth"),
    ("cmd.monitor.output.composite", "view.display.composite"),
    ("cmd.monitor.output.alpha", "view.display.alpha"),
    ("cmd.monitor.output.red", "view.display.red"),
    ("cmd.monitor.output.green", "view.display.green"),
    ("cmd.monitor.output.blue", "view.display.blue"),
    ("cmd.monitor.outputmulticam", "view.display.multicam"),
    ("cmd.monitor.outputaudiowaveform", "view.display.audioWaveform"),
    ("cmd.monitor.output.audio.video", "view.display.videoAndWaveform"),
    ("cmd.monitor.output.zoom.fit", "view.magnification.fit"),
    ("cmd.monitor.output.zoom.10", "view.magnification.10"),
    ("cmd.monitor.output.zoom.25", "view.magnification.25"),
    ("cmd.monitor.output.zoom.50", "view.magnification.50"),
    ("cmd.monitor.output.zoom.75", "view.magnification.75"),
    ("cmd.monitor.output.zoom.100", "view.magnification.100"),
    ("cmd.monitor.output.zoom.150", "view.magnification.150"),
    ("cmd.monitor.output.zoom.200", "view.magnification.200"),
    ("cmd.monitor.output.zoom.400", "view.magnification.400"),
    ("cmd.monitor.output.zoom.800", "view.magnification.800"),
    ("cmd.monitor.output.zoom.1600", "view.magnification.1600"),
    ("cmd.program.monitor.zoom.100", "view.programZoom100"),
    ("cmd.program.monitor.zoom.fit", "view.programZoomFit"),
    ("cmd.source.monitor.zoom.100", "view.sourceZoom100"),
    ("cmd.source.monitor.zoom.fit", "view.sourceZoomFit"),
    // Window
    ("cmd.toggle.maximize.focused.frame", "window.maximizeFrame"),
    ("cmd.toggle.maximize.frame", "window.maximizeFrameUnderCursor"),
    ("cmd.toggle.fullscreen.monitor", "window.toggleFullScreen"),
    ("cmd.select.next.panel", "window.nextPanel"),
    ("cmd.select.previous.panel", "window.prevPanel"),
    ("cmd.window.togglesourceprogram", "window.toggleMonitorFocus"),
    ("cmd.searchwindow.open", "window.openSearch"),
    ("cmd.select.find.box", "panel.selectFindBox"),
    ("cmd.window.workspace.revert", "window.workspace.reset"),
    ("cmd.window.workspace.save", "window.workspace.saveChanges"),
    ("cmd.window.workspace.new", "window.workspace.saveAs"),
    ("cmd.window.workspace.edit", "window.workspace.edit"),
    // Premiere's workspace slots, in the default workspace order
    ("cmd.window.user.workspace.0", "window.workspace.editing"),
    ("cmd.window.user.workspace.1", "window.workspace.assembly"),
    ("cmd.window.user.workspace.2", "window.workspace.color"),
    ("cmd.window.user.workspace.3", "window.workspace.effects"),
    ("cmd.window.user.workspace.4", "window.workspace.audio"),
    ("cmd.window.user.workspace.5", "window.workspace.captionsandgraphics"),
    ("cmd.window.user.workspace.6", "window.workspace.learning"),
    ("cmd.window.user.workspace.7", "window.workspace.review"),
    ("cmd.window.user.workspace.8", "window.workspace.allpanels"),
    ("cmd.window.effectcontrols", "window.panel.EffectControls"),
    ("cmd.window.effects", "window.panel.Effects"),
    ("cmd.window.events", "window.panel.Events"),
    ("cmd.window.history", "window.panel.History"),
    ("cmd.window.info", "window.panel.Info"),
    ("cmd.window.tools", "window.panel.Tools"),
    ("uif.window.Projects", "window.panel.Project"),
    ("uif.window.Source Monitors", "window.panel.Source"),
    ("uif.window.Timelines", "window.panel.Timeline"),
    ("uif.window.Program Monitors", "window.panel.Program"),
    ("uif.window.Effect Controls", "window.panel.EffectControls"),
    ("uif.window.Effects", "window.panel.Effects"),
    ("uif.window.Graphics", "window.panel.EssentialGraphics"),
    ("uif.window.Properties2", "window.panel.Properties"),
    ("uif.window.EssentialSound", "window.panel.EssentialSound"),
    ("uif.window.com.adobe.dva.text", "window.panel.Text"),
    ("uif.window.Audio Clip Mixer", "window.panel.AudioClipMixer"),
    ("uif.window.Audio Mixers", "window.panel.AudioTrackMixer"),
    ("uif.window.Audio Meter", "window.panel.AudioMeters"),
    ("uif.window.Media Browser", "window.panel.MediaBrowser"),
    ("uif.window.Metadata Editor", "window.panel.Metadata"),
    ("uif.window.MarkerList", "window.panel.Markers"),
    ("uif.window.History", "window.panel.History"),
    ("uif.window.Info", "window.panel.Info"),
    ("uif.window.Tools", "window.panel.Tools"),
    ("uif.window.Events", "window.panel.Events"),
    ("uif.window.Timecode", "window.panel.Timecode"),
    ("uif.window.Color", "window.panel.LumetriColor"),
    ("uif.window.Scopes", "window.panel.LumetriScopes"),
    ("uif.window.Reference Monitors", "window.panel.ReferenceMonitor"),
    ("uif.window.com.adobe.dva.progress", "window.panel.Progress"),
    // Project panel
    ("cmd.project.openinsource", "source.open"),
    ("cmd.project.toggle.view", "projectPanel.toggleView"),
    ("cmd.project.wingtip.view.list", "projectPanel.viewList"),
    ("cmd.project.wingtip.view.icon", "projectPanel.viewIcon"),
    ("cmd.project.enablehover", "projectPanel.hoverScrub"),
    ("cmd.project.wingtip.metadatadisplay", "projectPanel.metadataDisplay"),
    ("cmd.project.freeformview.options", "projectPanel.freeformOptions"),
    ("cmd.project.project.revealsystem", "projectPanel.revealProject"),
    ("cmd.file.project.revealsystem", "projectPanel.revealProject"),
    ("cmd.project.next.thumbnail.size.large", "projectPanel.thumbnailLarger"),
    ("cmd.project.previous.thumbnail.size.large", "projectPanel.thumbnailSmaller"),
    ("cmd.project.zoom.in", "projectPanel.thumbnailLarger"),
    ("cmd.project.zoom.out", "projectPanel.thumbnailSmaller"),
    ("cmd.project.move.up", "projectPanel.moveUp"),
    ("cmd.project.move.down", "projectPanel.moveDown"),
    // (sic: Premiere's own name)
    ("cmd.project.movel.eft", "projectPanel.moveLeft"),
    ("cmd.project.move.right", "projectPanel.moveRight"),
    ("cmd.project.move.home", "projectPanel.moveHome"),
    ("cmd.project.move.end", "projectPanel.moveEnd"),
    ("cmd.project.move.pageup", "projectPanel.movePageUp"),
    ("cmd.project.move.pagedown", "projectPanel.movePageDown"),
    ("cmd.project.moveextend.up", "projectPanel.extendUp"),
    ("cmd.project.moveextend.down", "projectPanel.extendDown"),
    ("cmd.project.moveextend.left", "projectPanel.extendLeft"),
    ("cmd.project.moveextend.right", "projectPanel.extendRight"),
    ("cmd.project.settings.general", "file.projectSettings.general"),
    ("cmd.project.settings.scratchdisks", "file.projectSettings.scratchDisks"),
    ("cmd.project.settings.ingestsettings", "project.ingestSettings"),
    ("cmd.mediabrowser.openinsourcemonitor", "mediaBrowser.openInSource"),
    ("cmd.mediabrowser.favorites.add", "mediaBrowser.favorite"),
    // Text panel (transcript)
    ("cmd.text.navigate.prev.word", "textPanel.prevWord"),
    ("cmd.text.navigate.next.word", "textPanel.nextWord"),
    ("cmd.text.select.prev.word", "textPanel.selectPrevWord"),
    ("cmd.text.select.next.word", "textPanel.selectNextWord"),
    ("cmd.text.navigate.prev.line", "textPanel.prevLine"),
    ("cmd.text.navigate.next.line", "textPanel.nextLine"),
    ("cmd.text.select.prev.line", "textPanel.selectPrevLine"),
    ("cmd.text.select.next.line", "textPanel.selectNextLine"),
    ("cmd.text.navigate.segment.start", "textPanel.segmentStart"),
    ("cmd.text.navigate.segment.end", "textPanel.segmentEnd"),
    ("cmd.text.select.segment.start", "textPanel.selectToSegmentStart"),
    ("cmd.text.select.segment.end", "textPanel.selectToSegmentEnd"),
    ("cmd.text.delete", "textPanel.delete"),
    ("cmd.text.ripple.delete", "textPanel.rippleDelete"),
    ("cmd.txt.navigate.programmonitor", "textPanel.showProgramTranscript"),
    // Help
    ("cmd.help.contents", "help.filmcraftHelp"),
    ("cmd.help.reveallogsfolder", "help.revealLogFiles"),
    ("cmd.help.systemcompatibilityreport", "help.systemCompatibilityReport"),
    ("cmd.help.about", "app.about"),
];

/// The FilmCraft command for a Premiere command in `context` (None: no counterpart).
pub fn map_command(context: &str, command: &str) -> Option<&'static str> {
    IN_CONTEXT.iter().find(|(c, p, _)| *c == context && *p == command).map(|e| e.2).or_else(|| COMMANDS.iter().find(|(p, _)| *p == command).map(|e| e.1))
}

/// A Premiere shortcut the import could not carry over.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Unmapped {
    /// Premiere's command name.
    pub command: String,
    /// Premiere context (`global`, `timeline`…).
    pub context: String,
    /// The key in FilmCraft notation (or the raw key code).
    pub keys: String,
    /// `no FilmCraft command`, `panel not in FilmCraft` or `unknown key`.
    pub reason: String,
}

/// What a `.kys` file becomes: bindings (FilmCraft notation), the (command, scope) pairs the file
/// leaves without a key, and the shortcuts that have no FilmCraft counterpart.
#[derive(Clone, Debug, Default)]
pub struct Converted {
    pub bindings: Vec<Binding>,
    pub unassigned: BTreeSet<(String, Option<String>)>,
    pub unmapped: Vec<Unmapped>,
    /// Shortcuts (items with a key) in the file.
    pub total: usize,
    /// Shortcuts carried over.
    pub mapped: usize,
}

/// Convert a parsed file; `known` says whether a FilmCraft command id exists in this session.
pub fn convert(file: &KysFile, known: impl Fn(&str) -> bool) -> Converted {
    let mut out = Converted::default();
    let mut unmapped: BTreeSet<Unmapped> = BTreeSet::new();
    for it in &file.items {
        let has_key = it.chord.is_some() || it.unknown_key.is_some();
        if has_key {
            out.total += 1;
        }
        let keys = it.chord.as_ref().map(Chord::canonical).or_else(|| it.unknown_key.map(|k| format!("key code {k}"))).unwrap_or_default();
        let miss = |reason: &str| Unmapped { command: it.command.clone(), context: it.context.clone(), keys: keys.clone(), reason: reason.into() };
        let Some(panel) = panel_of(&it.context) else {
            if has_key {
                unmapped.insert(miss("panel not in FilmCraft"));
            }
            continue;
        };
        let Some(id) = map_command(&it.context, &it.command).filter(|id| known(id)) else {
            if has_key {
                unmapped.insert(miss("no FilmCraft command"));
            }
            continue;
        };
        let panel = panel.map(str::to_string);
        match (&it.chord, it.unknown_key) {
            (Some(c), _) => {
                let b = Binding { command: id.to_string(), keys: c.canonical(), panel };
                out.mapped += 1;
                if !out.bindings.contains(&b) {
                    out.bindings.push(b);
                }
            }
            (None, Some(_)) => {
                unmapped.insert(miss("unknown key"));
            }
            (None, None) => {
                out.unassigned.insert((id.to_string(), panel));
            }
        }
    }
    // a command the file binds somewhere in a scope is not "unassigned" there
    let bound: BTreeSet<(String, Option<String>)> = out.bindings.iter().map(|b| (b.command.clone(), b.panel.clone())).collect();
    out.unassigned.retain(|k| !bound.contains(k));
    out.unmapped = unmapped.into_iter().collect();
    out
}

/// The preset name for a `.kys` file: `[Custom 2].kys` → `Premiere – Custom 2`.
pub fn preset_name(path: &std::path::Path) -> String {
    let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let stem = stem.trim().trim_start_matches('[').trim_end_matches(']').trim();
    if stem.is_empty() { "Premiere – Imported".into() } else { format!("Premiere – {stem}") }
}
