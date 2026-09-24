//! Labels Nemotron's text with Sortformer's speakers.
//!
//! Nemotron produces text per chunk; the diarizer produces speaker segments on
//! the same sample clock, but trails it by its buffer latency. So each chunk's
//! text is emitted at once, non-final, carrying the last resolved speaker so a
//! turn does not flicker, and is emitted again as final once the diarizer's
//! horizon has passed the chunk's end. Its speaker is then the one with the
//! most overlap with the chunk's sample range.
//!
//! Plain logic with no model, so every rule is unit-tested here. It knows
//! nothing of Sortformer's types and so builds without the `diarization`
//! feature; the backend then never constructs one.

use std::cmp::Reverse;
use std::collections::{BTreeMap, VecDeque};

use crate::asr::{SpeakerId, TranscriptUpdate};

/// How far before a chunk's first sample its text may describe, to allow for
/// encoder lookahead. Zero until measured on the two-speaker fixtures.
const LOOKBACK_SAMPLES: u64 = 0;

/// A stretch of one speaker's speech, `[start, end)` in samples, with the
/// diarizer's 0-based speaker id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Span {
    pub start: u64,
    pub end: u64,
    pub speaker: usize,
}

/// Text emitted non-final, waiting for the diarizer to cover its range.
struct Pending {
    segment_id: u64,
    start: u64,
    end: u64,
    text: String,
}

#[derive(Default)]
pub(super) struct Attribution {
    /// Speaker segments in arrival order, merged per speaker where they touch.
    /// Different speakers' spans may overlap: that is overlapped speech.
    timeline: Vec<Span>,
    /// In emission order, which is also the order finals must go out in.
    pending: VecDeque<Pending>,
    last_speaker: Option<usize>,
}

impl Attribution {
    /// Adds the diarizer's segments. A turn split at a window edge comes back
    /// as two touching segments; they are merged here, although majority
    /// overlap would give the same answer either way.
    #[cfg_attr(not(feature = "diarization"), allow(dead_code))]
    pub fn add_segments(&mut self, segments: impl IntoIterator<Item = Span>) {
        for s in segments.into_iter().filter(|s| s.end > s.start) {
            let previous = self
                .timeline
                .iter_mut()
                .rev()
                .find(|span| span.speaker == s.speaker);
            match previous {
                Some(span) if span.end >= s.start => span.end = span.end.max(s.end),
                _ => self.timeline.push(s),
            }
        }
    }

    /// Queues text for samples `[start, end)` and returns its non-final update.
    pub fn push(
        &mut self,
        segment_id: u64,
        start: u64,
        end: u64,
        text: String,
    ) -> TranscriptUpdate {
        let update = TranscriptUpdate {
            segment_id,
            is_final: false,
            text: text.clone(),
            speaker: self.last_speaker.map(speaker_label),
        };
        self.pending.push_back(Pending {
            segment_id,
            start,
            end,
            text,
        });
        update
    }

    /// Finalizes, in order, the pending text the diarizer has fully covered,
    /// then forgets spans ending before `keep_from` that no pending text can
    /// still need. `keep_from` is the earliest sample a later push may name,
    /// which a replay after a decoder reset makes older than the newest chunk.
    pub fn finalize_until(&mut self, horizon: u64, keep_from: u64) -> Vec<TranscriptUpdate> {
        let mut out = Vec::new();
        while self.pending.front().is_some_and(|p| p.end <= horizon) {
            let pending = self.pending.pop_front().expect("front checked above");
            out.push(self.finalize(pending));
        }
        let oldest_needed = self
            .pending
            .iter()
            .map(|p| p.start)
            .fold(keep_from, u64::min)
            .saturating_sub(LOOKBACK_SAMPLES);
        self.timeline.retain(|span| span.end > oldest_needed);
        out
    }

    /// End of session: finalizes everything with the best label available.
    pub fn finalize_all(&mut self) -> Vec<TranscriptUpdate> {
        self.finalize_until(u64::MAX, u64::MAX)
    }

    fn finalize(&mut self, pending: Pending) -> TranscriptUpdate {
        // No overlap means silence or pre-roll: the turn carries on.
        let speaker = self
            .majority_speaker(pending.start, pending.end)
            .or(self.last_speaker);
        if speaker.is_some() {
            self.last_speaker = speaker;
        }
        TranscriptUpdate {
            segment_id: pending.segment_id,
            is_final: true,
            text: pending.text,
            speaker: speaker.map(speaker_label),
        }
    }

    /// The speaker covering most of `[start − lookback, end)`. A tie keeps the
    /// current speaker, and otherwise goes to the lower id.
    fn majority_speaker(&self, start: u64, end: u64) -> Option<usize> {
        let from = start.saturating_sub(LOOKBACK_SAMPLES);
        let mut overlap: BTreeMap<usize, u64> = BTreeMap::new();
        for span in &self.timeline {
            let covered = end.min(span.end).saturating_sub(from.max(span.start));
            if covered > 0 {
                *overlap.entry(span.speaker).or_default() += covered;
            }
        }
        overlap
            .into_iter()
            .max_by_key(|&(speaker, covered)| {
                (
                    covered,
                    Some(speaker) == self.last_speaker,
                    Reverse(speaker),
                )
            })
            .map(|(speaker, _)| speaker)
    }
}

/// Sortformer's ids are 0-based; the UI shows `S1`, `S2`, …
fn speaker_label(speaker: usize) -> SpeakerId {
    SpeakerId((speaker + 1).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One 560 ms Nemotron chunk.
    const CHUNK: u64 = 8960;

    fn seg(speaker: usize, start: u64, end: u64) -> Span {
        Span {
            start,
            end,
            speaker,
        }
    }

    fn speaker(update: &TranscriptUpdate) -> Option<&str> {
        update.speaker.as_ref().map(|s| s.0.as_str())
    }

    #[test]
    fn majority_overlap_picks_the_speaker() {
        let mut a = Attribution::default();
        a.add_segments([seg(0, 0, 3000), seg(1, 3000, CHUNK)]);
        a.push(1, 0, CHUNK, "hello".into());

        let finals = a.finalize_until(CHUNK, CHUNK);

        assert_eq!(finals.len(), 1);
        assert!(finals[0].is_final);
        assert_eq!(finals[0].text, "hello");
        assert_eq!(speaker(&finals[0]), Some("2"));
    }

    #[test]
    fn overlapped_speech_sums_per_speaker() {
        let mut a = Attribution::default();
        // Speaker 0 in two stretches (5000 samples), speaker 1 throughout
        // one stretch of 4000.
        a.add_segments([
            seg(0, 0, 2500),
            seg(1, 2500, 6500),
            seg(0, 6460, CHUNK + 1000),
        ]);
        a.push(1, 0, CHUNK, "x".into());

        let finals = a.finalize_until(CHUNK, CHUNK);

        assert_eq!(speaker(&finals[0]), Some("1"));
    }

    #[test]
    fn a_window_edge_split_does_not_change_the_speaker() {
        let mut a = Attribution::default();
        // One turn by speaker 1, split at the 11 520-sample window edge, with
        // a brief blip of speaker 0 at the edge.
        a.add_segments([seg(1, 0, 11_520), seg(0, 11_520, 12_000)]);
        a.add_segments([seg(1, 11_520, 23_040)]);
        a.push(1, CHUNK, 2 * CHUNK, "straddles the edge".into());

        let finals = a.finalize_until(23_040, CHUNK);

        assert_eq!(speaker(&finals[0]), Some("2"));
        // The two halves were merged into one span.
        assert_eq!(
            a.timeline,
            vec![
                Span {
                    start: 0,
                    end: 23_040,
                    speaker: 1
                },
                Span {
                    start: 11_520,
                    end: 12_000,
                    speaker: 0
                },
            ]
        );
    }

    #[test]
    fn finalizes_only_past_the_horizon() {
        let mut a = Attribution::default();
        a.add_segments([seg(0, 0, 11_520)]);
        a.push(1, 0, CHUNK, "one".into());
        a.push(2, CHUNK, 2 * CHUNK, "two".into());

        let finals = a.finalize_until(11_520, 2 * CHUNK);
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].segment_id, 1);

        assert!(a.finalize_until(2 * CHUNK - 1, 2 * CHUNK).is_empty());

        a.add_segments([seg(0, 11_520, 23_040)]);
        let finals = a.finalize_until(23_040, 2 * CHUNK);
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].segment_id, 2);
        assert_eq!(speaker(&finals[0]), Some("1"));
    }

    #[test]
    fn provisional_label_is_the_last_resolved_speaker() {
        let mut a = Attribution::default();
        let first = a.push(1, 0, CHUNK, "one".into());
        assert!(!first.is_final);
        assert_eq!(first.text, "one");
        assert_eq!(speaker(&first), None);

        a.add_segments([seg(0, 0, CHUNK)]);
        a.finalize_until(CHUNK, CHUNK);

        // Speaker 1 is already talking in the diarizer's future, but only a
        // resolved label is ever shown provisionally.
        let second = a.push(2, CHUNK, 2 * CHUNK, "two".into());
        assert_eq!(speaker(&second), Some("1"));
    }

    #[test]
    fn no_overlap_inherits_the_previous_speaker() {
        let mut a = Attribution::default();
        a.add_segments([seg(1, 0, CHUNK)]);
        a.push(1, 0, CHUNK, "speech".into());
        // Zero padding after speech end: the diarizer reports nobody.
        a.push(2, CHUNK, 2 * CHUNK, "tail".into());

        let finals = a.finalize_until(2 * CHUNK, 2 * CHUNK);

        assert_eq!(speaker(&finals[0]), Some("2"));
        assert_eq!(speaker(&finals[1]), Some("2"));
    }

    #[test]
    fn nothing_known_yet_leaves_the_speaker_unset() {
        let mut a = Attribution::default();
        a.push(1, 0, CHUNK, "early".into());

        let finals = a.finalize_until(CHUNK, CHUNK);

        assert_eq!(speaker(&finals[0]), None);
    }

    #[test]
    fn a_tie_keeps_the_current_speaker() {
        let mut a = Attribution::default();
        a.add_segments([seg(1, 0, CHUNK)]);
        a.push(1, 0, CHUNK, "a".into());
        a.finalize_until(CHUNK, CHUNK);

        let half = CHUNK / 2;
        a.add_segments([seg(0, CHUNK, CHUNK + half), seg(1, CHUNK + half, 2 * CHUNK)]);
        a.push(2, CHUNK, 2 * CHUNK, "b".into());
        let finals = a.finalize_until(2 * CHUNK, 2 * CHUNK);

        assert_eq!(speaker(&finals[0]), Some("2"));
    }

    #[test]
    fn replayed_text_is_labelled_by_its_original_range() {
        let mut a = Attribution::default();
        // Speaker 0 for chunks 0 and 1, then speaker 1 from chunk 2.
        a.add_segments([seg(0, 0, 2 * CHUNK), seg(1, 2 * CHUNK, 4 * CHUNK)]);
        a.push(1, 0, CHUNK, "a".into());
        a.finalize_until(CHUNK, CHUNK);

        // Chunk 2 is live; then a decoder reset replays chunk 1, whose text
        // arrives later but describes earlier audio.
        a.push(2, 2 * CHUNK, 3 * CHUNK, "live".into());
        a.push(3, CHUNK, 2 * CHUNK, "replayed".into());

        let finals = a.finalize_until(4 * CHUNK, CHUNK);

        let ids: Vec<u64> = finals.iter().map(|u| u.segment_id).collect();
        assert_eq!(ids, vec![2, 3]);
        assert_eq!(speaker(&finals[0]), Some("2"));
        assert_eq!(speaker(&finals[1]), Some("1"));
    }

    #[test]
    fn finalize_all_flushes_everything_in_order() {
        let mut a = Attribution::default();
        a.add_segments([seg(0, 0, CHUNK)]);
        a.push(1, 0, CHUNK, "one".into());
        a.push(2, CHUNK, 2 * CHUNK, "two".into());

        let finals = a.finalize_all();

        assert_eq!(finals.len(), 2);
        assert!(finals.iter().all(|u| u.is_final));
        assert_eq!(speaker(&finals[1]), Some("1"));
        assert!(a.pending.is_empty());
        assert!(a.timeline.is_empty());
    }

    #[test]
    fn spans_are_kept_while_pending_or_replayable_text_needs_them() {
        let mut a = Attribution::default();
        a.add_segments([seg(0, 0, CHUNK), seg(1, CHUNK, 2 * CHUNK)]);
        a.push(1, CHUNK, 2 * CHUNK, "waiting".into());

        // Replay could still reach back to sample 0.
        a.finalize_until(CHUNK, 0);
        assert_eq!(a.timeline.len(), 2);

        // Nothing can reach back past CHUNK any more.
        a.finalize_until(CHUNK, 3 * CHUNK);
        assert_eq!(a.timeline.len(), 1);
        assert_eq!(a.timeline[0].speaker, 1);
    }
}
