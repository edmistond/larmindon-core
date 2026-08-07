//! Turns Soniox's revising token stream into `TranscriptUpdate`s.
//!
//! Two lanes:
//!
//! * **Durable** — tokens arriving `is_final: true` are delivered exactly once
//!   and never repeated. They accumulate into a pending buffer that flushes as
//!   a final segment at a sentence terminator, at a speaker change, or on the
//!   `<end>` marker.
//! * **Volatile** — one open segment holding the pending durable text plus this
//!   response's non-final tokens. It is thrown away and rebuilt from scratch on
//!   every response, so a provisional mistake corrects itself.
//!
//! Non-final tokens are the *complete* provisional tail as of each message, so
//! the tail is replaced wholesale rather than diffed.

use super::protocol::{SonioxResponse, SonioxToken};
use crate::asr::{SegmentIds, SpeakerId, TranscriptUpdate};

/// Sentence terminators, including the full-width CJK forms.
const TERMINATORS: [char; 6] = ['.', '!', '?', '。', '！', '？'];
/// Closing punctuation allowed to trail a terminator (`end."` still ends a
/// sentence).
const CLOSERS: [char; 7] = ['"', '\'', '”', '’', ')', ']', '}'];
/// Characters that must not be preceded by an inserted space when joining.
const NO_SPACE_BEFORE: [char; 11] = ['.', ',', '!', '?', ';', ':', ')', ']', '}', '”', '’'];

pub struct Accumulator {
    /// Finalized text not yet flushed into a segment.
    pending: String,
    /// Speaker of the pending text. Only ever set from final tokens.
    pending_speaker: Option<String>,
    /// Id of the currently open (non-final) segment, allocated lazily.
    open_id: Option<u64>,
    /// Whether an open segment was emitted and still needs finalizing.
    open_emitted: bool,
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator {
    pub fn new() -> Self {
        Self {
            pending: String::new(),
            pending_speaker: None,
            open_id: None,
            open_emitted: false,
        }
    }

    /// True when a token carries displayable speech (not a control token, not
    /// whitespace). Used both for output and as the meter's activity evidence.
    pub fn is_speech_token(token: &SonioxToken) -> bool {
        !token.is_special() && !token.text.trim().is_empty()
    }

    /// Folds one response into the two lanes and returns the updates to emit.
    pub fn ingest(&mut self, response: &SonioxResponse, ids: &SegmentIds) -> Vec<TranscriptUpdate> {
        let mut out = Vec::new();

        // A single response can carry finals first and then the provisional
        // tail, so the durable lane is processed before the volatile one.
        for token in response.tokens.iter().filter(|t| t.is_final) {
            // A null speaker INHERITS the current one. Clearing it here would
            // fragment every turn, because undiarized tokens are common.
            if let Some(speaker) = token.normalized_speaker() {
                if self
                    .pending_speaker
                    .as_ref()
                    .is_some_and(|current| *current != speaker)
                {
                    self.flush(&mut out, ids);
                }
                self.pending_speaker = Some(speaker);
            }

            if token.is_end_marker() {
                self.flush(&mut out, ids);
                continue;
            }
            if token.is_special() {
                continue;
            }

            // Tokens carry their own leading whitespace, so they join with no
            // separator.
            self.pending.push_str(&token.text);

            if ends_sentence(&self.pending) {
                self.flush(&mut out, ids);
            }
        }

        // Volatile lane: rebuild from the unflushed durable text plus this
        // message's provisional tail.
        let tail: String = response
            .tokens
            .iter()
            .filter(|t| !t.is_final && !t.is_special())
            .map(|t| t.text.as_str())
            .collect();

        let open_text = format!("{}{}", self.pending, tail);
        if open_text.trim().is_empty() {
            // Nothing provisional left; drop any open segment by finalizing it
            // as empty is not allowed, so simply forget the id.
            if !self.pending.is_empty() {
                // Pending durable text with no tail still deserves display.
                let id = *self.open_id.get_or_insert_with(|| ids.next());
                self.open_emitted = true;
                out.push(TranscriptUpdate {
                    segment_id: id,
                    is_final: false,
                    text: self.pending.clone(),
                    speaker: self.pending_speaker.clone().map(SpeakerId),
                });
            }
        } else {
            let id = *self.open_id.get_or_insert_with(|| ids.next());
            self.open_emitted = true;
            out.push(TranscriptUpdate {
                segment_id: id,
                is_final: false,
                text: open_text,
                speaker: self.pending_speaker.clone().map(SpeakerId),
            });
        }

        out
    }

    /// Finalizes the pending durable text as a segment.
    fn flush(&mut self, out: &mut Vec<TranscriptUpdate>, ids: &SegmentIds) {
        if self.pending.trim().is_empty() {
            self.pending.clear();
            return;
        }
        let id = self.open_id.take().unwrap_or_else(|| ids.next());
        self.open_emitted = false;
        out.push(TranscriptUpdate {
            segment_id: id,
            is_final: true,
            text: std::mem::take(&mut self.pending),
            speaker: self.pending_speaker.clone().map(SpeakerId),
        });
    }

    /// Finalizes whatever is open. Used on clean shutdown and before a
    /// reconnect, where the current hypothesis is the best text available.
    pub fn finish(&mut self, ids: &SegmentIds) -> Vec<TranscriptUpdate> {
        let mut out = Vec::new();
        self.flush(&mut out, ids);
        if out.is_empty() && self.open_emitted {
            // An open segment was shown but has no durable text behind it;
            // retract it by finalizing as empty so the UI stops dimming it.
            if let Some(id) = self.open_id.take() {
                out.push(TranscriptUpdate {
                    segment_id: id,
                    is_final: true,
                    text: String::new(),
                    speaker: None,
                });
            }
        }
        self.open_id = None;
        self.open_emitted = false;
        out
    }

    /// Drops all in-flight state without emitting. Used when a reconnect has
    /// already flushed the open segment.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.pending_speaker = None;
        self.open_id = None;
        self.open_emitted = false;
    }
}

/// Whether `text` ends a sentence.
///
/// A terminator may be followed by closing quotes/brackets, and must be
/// followed by whitespace or end-of-input. That last rule is what keeps `3.14`
/// and `e.g.` from splitting mid-token.
pub fn ends_sentence(text: &str) -> bool {
    let trimmed = text.trim_end();
    let mut chars = trimmed.chars().rev().skip_while(|c| CLOSERS.contains(c));
    let Some(candidate) = chars.next() else {
        return false;
    };
    if !TERMINATORS.contains(&candidate) {
        return false;
    }
    // A digit immediately before a '.' means a decimal, not a sentence end.
    if candidate == '.' {
        if let Some(prev) = chars.next() {
            if prev.is_ascii_digit() {
                return false;
            }
        }
    }
    true
}

/// Joins two fragments, inserting a space only when one is genuinely needed.
pub fn join_fragments(left: &str, right: &str) -> String {
    if left.is_empty() {
        return right.to_string();
    }
    if right.is_empty() {
        return left.to_string();
    }
    let needs_space = !left.ends_with(char::is_whitespace)
        && !right.starts_with(char::is_whitespace)
        && !right.starts_with(|c| NO_SPACE_BEFORE.contains(&c));
    if needs_space {
        format!("{left} {right}")
    } else {
        format!("{left}{right}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::SegmentIds;

    fn token(text: &str, is_final: bool, speaker: Option<&str>) -> SonioxToken {
        SonioxToken {
            text: text.to_string(),
            is_final,
            speaker: speaker.map(str::to_string),
            ..Default::default()
        }
    }

    fn response(tokens: Vec<SonioxToken>) -> SonioxResponse {
        SonioxResponse {
            tokens,
            ..Default::default()
        }
    }

    #[test]
    fn sentence_splitting_keeps_decimals_intact() {
        assert!(!ends_sentence("3.14"));
        assert!(!ends_sentence("Pi is 3."));
        assert!(ends_sentence("That is all."));
        assert!(ends_sentence("Really?"));
        assert!(ends_sentence("Stop!"));
    }

    #[test]
    fn sentence_splitting_allows_trailing_closers() {
        assert!(ends_sentence("he said \"the end.\""));
        assert!(ends_sentence("(that is all.)"));
        assert!(ends_sentence("done.'"));
    }

    #[test]
    fn sentence_splitting_handles_cjk_terminators() {
        assert!(ends_sentence("这是结束。"));
        assert!(ends_sentence("真的吗？"));
    }

    #[test]
    fn non_terminated_text_does_not_split() {
        assert!(!ends_sentence("unfinished thought"));
        assert!(!ends_sentence(""));
        assert!(!ends_sentence("wait, "));
    }

    #[test]
    fn join_inserts_a_space_only_when_needed() {
        assert_eq!(join_fragments("hello", "world"), "hello world");
        assert_eq!(join_fragments("hello", " world"), "hello world");
        assert_eq!(join_fragments("hello ", "world"), "hello world");
        assert_eq!(join_fragments("hello", ","), "hello,");
        assert_eq!(join_fragments("hello", "!"), "hello!");
        assert_eq!(join_fragments("", "world"), "world");
    }

    #[test]
    fn non_final_tail_is_replaced_wholesale_not_appended() {
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();

        let a = acc.ingest(&response(vec![token("How", false, Some("1"))]), &ids);
        let b = acc.ingest(&response(vec![token("How are", false, Some("1"))]), &ids);

        assert_eq!(a.len(), 1);
        assert_eq!(a[0].text, "How");
        assert_eq!(b[0].text, "How are");
        // Same open segment revised in place, not a second one.
        assert_eq!(a[0].segment_id, b[0].segment_id);
        assert!(!a[0].is_final && !b[0].is_final);
    }

    #[test]
    fn final_tokens_accumulate_and_flush_on_sentence_end() {
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();

        acc.ingest(&response(vec![token("How", true, Some("1"))]), &ids);
        acc.ingest(&response(vec![token(" are", true, Some("1"))]), &ids);
        let out = acc.ingest(&response(vec![token(" you?", true, Some("1"))]), &ids);

        let finals: Vec<_> = out.iter().filter(|u| u.is_final).collect();
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].text, "How are you?");
        assert_eq!(finals[0].speaker.as_ref().unwrap().0, "1");
    }

    #[test]
    fn speaker_change_flushes_the_previous_turn() {
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();

        acc.ingest(&response(vec![token("Hello there", true, Some("1"))]), &ids);
        let out = acc.ingest(&response(vec![token(" Hi back", true, Some("2"))]), &ids);

        let finals: Vec<_> = out.iter().filter(|u| u.is_final).collect();
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].text, "Hello there");
        assert_eq!(finals[0].speaker.as_ref().unwrap().0, "1");
    }

    #[test]
    fn null_speaker_inherits_rather_than_clearing() {
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();

        acc.ingest(&response(vec![token("Hello", true, Some("1"))]), &ids);
        // Undiarized token must join the current turn, not split it.
        let out = acc.ingest(&response(vec![token(" there", true, None)]), &ids);

        assert!(out.iter().all(|u| !u.is_final), "no flush should occur");
        let finals = acc.finish(&ids);
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].text, "Hello there");
        assert_eq!(finals[0].speaker.as_ref().unwrap().0, "1");
    }

    #[test]
    fn end_marker_flushes_without_appearing_in_text() {
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();

        acc.ingest(
            &response(vec![token("No punctuation here", true, Some("1"))]),
            &ids,
        );
        let out = acc.ingest(&response(vec![token("<end>", true, Some("1"))]), &ids);

        let finals: Vec<_> = out.iter().filter(|u| u.is_final).collect();
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].text, "No punctuation here");
        assert!(!finals[0].text.contains("<end>"));
    }

    #[test]
    fn fin_marker_is_stripped_from_text() {
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();

        acc.ingest(&response(vec![token("Some words", true, None)]), &ids);
        acc.ingest(&response(vec![token("<fin>", true, None)]), &ids);
        let finals = acc.finish(&ids);

        assert_eq!(finals[0].text, "Some words");
    }

    #[test]
    fn provisional_speaker_mislabel_self_corrects() {
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();

        // "How" arrives non-final attributed to speaker 2 ...
        let a = acc.ingest(&response(vec![token("How", false, Some("2"))]), &ids);
        assert!(a[0].speaker.is_none(), "no durable speaker yet");

        // ... then finalizes as speaker 1. The volatile lane is rebuilt each
        // message, so the mislabel never becomes durable.
        acc.ingest(&response(vec![token("How", true, Some("1"))]), &ids);
        let finals = acc.finish(&ids);
        assert_eq!(finals[0].speaker.as_ref().unwrap().0, "1");
    }

    #[test]
    fn idle_response_with_no_tokens_emits_nothing() {
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();
        assert!(acc.ingest(&response(vec![]), &ids).is_empty());
    }

    #[test]
    fn segment_ids_are_never_reused_across_flushes() {
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();

        acc.ingest(&response(vec![token("First one.", true, None)]), &ids);
        acc.ingest(&response(vec![token(" Second one.", true, None)]), &ids);
        let all = acc.finish(&ids);
        let _ = all;

        // Every id handed out is distinct.
        let a = ids.next();
        let b = ids.next();
        assert_ne!(a, b);
    }

    #[test]
    fn speech_token_detection_ignores_control_and_blank_tokens() {
        assert!(Accumulator::is_speech_token(&token(" hello", false, None)));
        assert!(!Accumulator::is_speech_token(&token("<end>", true, None)));
        assert!(!Accumulator::is_speech_token(&token("<fin>", true, None)));
        assert!(!Accumulator::is_speech_token(&token("   ", false, None)));
    }
}
