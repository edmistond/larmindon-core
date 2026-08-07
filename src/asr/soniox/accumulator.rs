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
    /// The latest provisional tail, replaced wholesale by every response.
    ///
    /// Retained rather than computed and dropped, because at teardown or before
    /// a reconnect it is the best text the open segment will ever have — no
    /// further revision is coming. Dropping it truncates the last utterance,
    /// which is exactly what a speaker notices.
    tail: String,
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
            tail: String::new(),
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
        // message's provisional tail. The tail replaces its predecessor
        // wholesale — it is the complete hypothesis as of this response, not a
        // delta — so there is no prefix-diffing to do.
        self.tail = response
            .tokens
            .iter()
            .filter(|t| !t.is_final && !t.is_special())
            .map(|t| t.text.as_str())
            .collect();

        let open_text = format!("{}{}", self.pending, self.tail);
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
    ///
    /// Promoting the provisional tail is the whole point. The service only
    /// finalizes tokens once it is confident, which in practice means the last
    /// utterance of a session is often still provisional when the user presses
    /// Stop — there is no trailing silence to trigger finalization. Flushing
    /// only the durable lane truncates that utterance mid-sentence.
    pub fn finish(&mut self, ids: &SegmentIds) -> Vec<TranscriptUpdate> {
        // Nothing further is coming, so the tail stops being provisional.
        if !self.tail.trim().is_empty() {
            let tail = std::mem::take(&mut self.tail);
            self.pending.push_str(&tail);
        }
        self.tail.clear();

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
        self.tail.clear();
        self.pending_speaker = None;
        self.open_id = None;
        self.open_emitted = false;
    }
}

/// Whether `text` ends a sentence.
///
/// A terminator may be followed by closing quotes/brackets. Note that this only
/// ever sees text *up to* the candidate — there is no lookahead, because tokens
/// arrive one at a time — so "is the next character whitespace?" is not a
/// question this can ask. Everything it rejects, it rejects on what precedes
/// the dot.
///
/// Two rejections, both observed rather than imagined:
///
/// * a digit before `.` is a decimal (`3.14`);
/// * a lone letter before `.` is an abbreviation or an initial (`e.g.`, `U.S.`,
///   `J. R. R.`), never a word ending a sentence.
///
/// Known limitation: multi-letter abbreviations (`etc.`, `Mr.`, `vs.`) still
/// split. Fixing those needs a word list, which is locale-specific and brittle;
/// the cost here is a spurious segment boundary, not lost text.
pub fn ends_sentence(text: &str) -> bool {
    let trimmed = text.trim_end();
    let mut chars = trimmed.chars().rev().skip_while(|c| CLOSERS.contains(c));
    let Some(candidate) = chars.next() else {
        return false;
    };
    if !TERMINATORS.contains(&candidate) {
        return false;
    }
    if candidate == '.' {
        match chars.next() {
            Some(prev) if prev.is_ascii_digit() => return false,
            Some(prev) if prev.is_alphabetic() => {
                // A single letter, i.e. one preceded by a boundary or by
                // another abbreviation dot. "put." has 'u' before 't', so it
                // splits; "e." and the 'g' of "e.g." do not.
                let boundary = chars
                    .next()
                    .is_none_or(|before| before.is_whitespace() || before == '.');
                if boundary {
                    return false;
                }
            }
            _ => {}
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
    fn sentence_splitting_keeps_abbreviations_and_initials_intact() {
        // Observed live: " ... the renewal rate stabilizes, e." finalized as one
        // segment and "g., whether the enterprise accounts stay put." as the
        // next, so a segment began mid-word.
        assert!(!ends_sentence("whether the rate stabilizes, e."));
        assert!(!ends_sentence("whether the rate stabilizes, e.g."));
        assert!(!ends_sentence("based in the U."));
        assert!(!ends_sentence("based in the U.S."));
        assert!(!ends_sentence("J."));
        assert!(!ends_sentence("written by J. R. R."));
        // A real sentence end still splits: 't' is preceded by 'u', not a
        // boundary.
        assert!(ends_sentence("whether the accounts stay put."));
        assert!(ends_sentence("That is all."));
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
    fn finish_promotes_the_provisional_tail_instead_of_truncating() {
        // Observed live: a session's last utterance is usually still
        // provisional at Stop, because there is no trailing silence to make the
        // service finalize it. Flushing only the durable lane produced
        // "Revenue came in at" and threw away "3.14 million, which is about 4%
        // under plan."
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();

        acc.ingest(
            &response(vec![
                token("Revenue came in at", true, Some("2")),
                token(" 3.14 million, which is about 4% under plan.", false, None),
            ]),
            &ids,
        );

        let finals = acc.finish(&ids);
        assert_eq!(finals.len(), 1);
        assert!(finals[0].is_final);
        assert_eq!(
            finals[0].text,
            "Revenue came in at 3.14 million, which is about 4% under plan."
        );
    }

    #[test]
    fn finish_promotes_a_tail_with_no_durable_text_behind_it() {
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();

        acc.ingest(
            &response(vec![token("Nothing final yet", false, None)]),
            &ids,
        );

        let finals = acc.finish(&ids);
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].text, "Nothing final yet");
        assert!(finals[0].is_final);
    }

    #[test]
    fn a_stale_tail_does_not_survive_into_the_next_segment() {
        // The tail is replaced wholesale by every response, so a flush must not
        // leave the previous hypothesis behind to be promoted later.
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();

        acc.ingest(&response(vec![token("First guess", false, None)]), &ids);
        // The guess finalizes and the sentence closes; no tail remains.
        acc.ingest(&response(vec![token("First guess.", true, None)]), &ids);

        let finals = acc.finish(&ids);
        assert!(
            finals.is_empty(),
            "nothing should remain after a clean flush, got {finals:?}"
        );
    }

    #[test]
    fn reset_discards_the_tail_so_a_reconnect_does_not_replay_it() {
        let ids = SegmentIds::new();
        let mut acc = Accumulator::new();

        acc.ingest(&response(vec![token("half a thought", false, None)]), &ids);
        // A reconnect flushes first, then resets.
        let _ = acc.finish(&ids);
        acc.reset();

        assert!(acc.finish(&ids).is_empty());
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
