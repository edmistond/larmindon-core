//! On-device streaming ASR via `parakeet-rs` (Nemotron).
//!
//! Append-only: every segment is emitted already final, each id once, and the
//! speaker is always `None`. Moved out of `audio_engine.rs` unchanged — the
//! chunking, the replay buffer and all four decoder-reset mechanisms behave
//! exactly as they did inline.

use std::collections::VecDeque;
use std::time::Instant;

#[cfg(any(feature = "webgpu", feature = "directml", feature = "migraphx"))]
use parakeet_rs::ExecutionProvider;
use parakeet_rs::{ExecutionConfig, Nemotron};

use super::{
    ActivitySource, AsrBackend, AsrCapabilities, AsrContext, AsrError, AudioGating, SessionContext,
    TranscriptUpdate,
};
use crate::diag::{
    AsrErrorRow, MidSpeechResetRow, PunctuationResetRow, SpeechEndRow, TranscribeRow,
};
use crate::settings::{self, Settings};
use crate::vad::VadState;

/// Recent chunks kept so they can be re-run through a freshly reset decoder.
struct ReplayBuffer {
    chunks: VecDeque<Vec<f32>>,
    max_chunks: usize,
}

struct PendingSpeechEndReset {
    reset_after_samples: usize,
    uptime_ms: i64,
    speech_duration_ms: f64,
}

impl ReplayBuffer {
    fn new(max_chunks: usize) -> Self {
        Self {
            chunks: VecDeque::with_capacity(max_chunks),
            max_chunks,
        }
    }

    fn update_capacity(&mut self, max_chunks: usize) {
        self.max_chunks = max_chunks;
        self.truncate_to_capacity();
    }

    fn push(&mut self, chunk: &[f32]) {
        if self.max_chunks == 0 {
            return;
        }
        self.chunks.push_back(chunk.to_vec());
        self.truncate_to_capacity();
    }

    fn clear(&mut self) {
        self.chunks.clear();
    }

    fn snapshot(&self) -> Vec<Vec<f32>> {
        self.chunks.iter().cloned().collect()
    }

    fn truncate_to_capacity(&mut self) {
        while self.chunks.len() > self.max_chunks {
            self.chunks.pop_front();
        }
    }
}

pub struct NemotronBackend {
    model: Option<Nemotron>,
    asr_buffer: Vec<f32>,
    chunk_size: usize,
    chunk_num: u64,
    consecutive_empty: u32,
    chunks_since_decoder_reset: u64,
    punctuation_reset_enabled: bool,
    empty_reset_threshold: u32,
    replay_buffer: ReplayBuffer,
    pending_speech_end_resets: VecDeque<PendingSpeechEndReset>,
}

impl NemotronBackend {
    pub fn new() -> Self {
        Self {
            model: None,
            asr_buffer: Vec::new(),
            chunk_size: 0,
            chunk_num: 0,
            consecutive_empty: 0,
            chunks_since_decoder_reset: 0,
            punctuation_reset_enabled: true,
            empty_reset_threshold: 1,
            replay_buffer: ReplayBuffer::new(0),
            pending_speech_end_resets: VecDeque::new(),
        }
    }
}

impl Default for NemotronBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AsrBackend for NemotronBackend {
    fn id(&self) -> &'static str {
        "nemotron"
    }

    fn capabilities(&self) -> AsrCapabilities {
        AsrCapabilities {
            revises_output: false,
            diarization: false,
            audio_gating: AudioGating::VadGated,
            activity_source: ActivitySource::Vad,
        }
    }

    fn is_cacheable(&self) -> bool {
        true
    }

    fn begin_session(&mut self, ctx: &SessionContext) -> Result<(), AsrError> {
        let settings = ctx.settings;

        // A model surviving from a previous session is reset rather than
        // reloaded, exactly as the engine's model cache did inline.
        if let Some(model) = self.model.as_mut() {
            println!("Using cached Nemotron model (skipping reload)");
            model.reset();
        } else {
            let model_path = settings::expand_tilde(&settings.model_path);
            println!(
                "Loading Nemotron model from {} (intra_threads={}, inter_threads={})...",
                model_path.display(),
                settings.intra_threads,
                settings.inter_threads
            );
            #[allow(unused_mut)]
            let mut model_config = ExecutionConfig::new()
                .with_intra_threads(settings.intra_threads)
                .with_inter_threads(settings.inter_threads);

            #[cfg(feature = "webgpu")]
            {
                println!("WebGPU feature enabled — using WebGPU (Metal) execution provider");
                model_config = model_config.with_execution_provider(ExecutionProvider::WebGPU);
            }

            #[cfg(feature = "directml")]
            {
                println!("DirectML feature enabled - using DirectML execution provider");
                model_config = model_config.with_execution_provider(ExecutionProvider::DirectML);
            }

            let m = Nemotron::from_pretrained(&model_path, Some(model_config))
                .map_err(|e| AsrError::Fatal(format!("Failed to load Nemotron model: {e}")))?;
            println!("Model loaded.");
            self.model = Some(m);
        }

        self.chunk_size = settings::chunk_ms_to_samples(settings.chunk_ms);
        self.punctuation_reset_enabled = settings.punctuation_reset;
        self.empty_reset_threshold = settings.empty_reset_threshold;
        self.replay_buffer = ReplayBuffer::new(settings.empty_reset_threshold as usize);
        self.pending_speech_end_resets.clear();
        self.asr_buffer = Vec::with_capacity(self.chunk_size * 2);
        self.chunk_num = 0;
        self.consecutive_empty = 0;
        self.chunks_since_decoder_reset = 0;
        Ok(())
    }

    fn push_audio(&mut self, samples: &[f32]) {
        self.asr_buffer.extend_from_slice(samples);
    }

    fn mark_speech_start(&mut self) {
        self.consecutive_empty = 0;
        self.replay_buffer.clear();
    }

    fn mark_speech_end(&mut self, uptime_ms: i64, speech_duration_ms: f64) {
        pad_to_chunk_boundary(&mut self.asr_buffer, self.chunk_size);
        self.pending_speech_end_resets
            .push_back(PendingSpeechEndReset {
                reset_after_samples: self.asr_buffer.len(),
                uptime_ms,
                speech_duration_ms,
            });
    }

    fn chunks_processed(&self) -> u64 {
        self.chunk_num
    }

    fn update_settings(&mut self, settings: &Settings) {
        self.punctuation_reset_enabled = settings.punctuation_reset;
        self.empty_reset_threshold = settings.empty_reset_threshold;
        self.replay_buffer
            .update_capacity(self.empty_reset_threshold as usize);
    }

    /// Stopping mid-utterance discards whatever sits in `asr_buffer` below one
    /// chunk. That is existing behaviour and is deliberately preserved —
    /// flushing the tail here would change Nemotron's output.
    fn end_session(&mut self, _ctx: &SessionContext) -> Result<Vec<TranscriptUpdate>, AsrError> {
        Ok(Vec::new())
    }

    fn process(&mut self, ctx: &AsrContext) -> Result<Vec<TranscriptUpdate>, AsrError> {
        let Self {
            model,
            asr_buffer,
            chunk_size,
            chunk_num,
            consecutive_empty,
            chunks_since_decoder_reset,
            punctuation_reset_enabled,
            empty_reset_threshold,
            replay_buffer,
            pending_speech_end_resets,
        } = self;

        let model = model
            .as_mut()
            .ok_or_else(|| AsrError::Fatal("Nemotron model not loaded".to_string()))?;
        let chunk_size = *chunk_size;
        let punctuation_reset_enabled = *punctuation_reset_enabled;
        let empty_reset_threshold = *empty_reset_threshold;

        let diag = ctx.session.diag;
        let ids = ctx.session.ids;
        let stats = ctx.stats;
        let iter_start = stats.iter_start;
        let drain_count = stats.drain_samples;
        let drain_audio_ms = stats.drain_audio_ms;
        let vad_ms = stats.vad_ms;
        let resample_ms = stats.resample_ms;
        let vad_state_str = match stats.vad_state {
            VadState::Silence => "silence",
            VadState::Speech => "speech",
        };

        let mut out: Vec<TranscriptUpdate> = Vec::new();

        let mut asr_consumed = 0;
        while asr_buffer.len().saturating_sub(asr_consumed) >= chunk_size {
            let chunk = &asr_buffer[asr_consumed..asr_consumed + chunk_size];
            asr_consumed += chunk_size;
            let vad_is_speech = stats.vad_state == VadState::Speech;
            if vad_is_speech {
                replay_buffer.push(chunk);
            }
            *chunks_since_decoder_reset += 1;
            let infer_start = Instant::now();
            match model.transcribe_chunk(chunk) {
                Ok(text) => {
                    let infer_ms = infer_start.elapsed().as_millis() as i64;
                    *chunk_num += 1;
                    let is_empty = text.is_empty();
                    let preview = text_preview(&text, 200);

                    if is_empty && vad_is_speech {
                        *consecutive_empty += 1;
                    } else {
                        *consecutive_empty = 0;
                    }

                    let iteration_ms = iter_start.elapsed().as_millis() as i64;
                    diag.transcribe(&TranscribeRow {
                        uptime_ms: ctx.session.uptime_ms(),
                        chunk_num: *chunk_num,
                        inference_ms: infer_ms,
                        drain_samples: drain_count,
                        drain_audio_ms,
                        asr_buf_len: asr_buffer.len().saturating_sub(asr_consumed),
                        is_empty,
                        preview: &preview,
                        vad_state: vad_state_str,
                        vad_ms,
                        resample_ms,
                        iteration_ms,
                        chunk_source: "live",
                    });

                    // Punctuation-based decoder reset
                    if punctuation_reset_enabled
                        && !is_empty
                        && ends_with_sentence_punctuation(&text)
                    {
                        diag.punctuation_reset(&PunctuationResetRow {
                            uptime_ms: ctx.session.uptime_ms(),
                            consecutive_empty: *consecutive_empty,
                            chunks_since_decoder_reset: *chunks_since_decoder_reset,
                        });
                        model.reset();
                        *chunks_since_decoder_reset = 0;
                        *consecutive_empty = 0;
                        replay_buffer.clear();
                    }

                    if !is_empty {
                        out.push(TranscriptUpdate {
                            segment_id: ids.next(),
                            is_final: true,
                            text,
                            speaker: None,
                        });
                    }

                    // Mid-speech reset heuristic
                    if *consecutive_empty >= empty_reset_threshold && vad_is_speech {
                        let consecutive_empty_at_reset = *consecutive_empty;
                        let reset_uptime = ctx.session.uptime_ms();
                        let chunks_at_reset = *chunks_since_decoder_reset;
                        let replay_chunks = replay_buffer.snapshot();
                        let replay_chunk_count = replay_chunks.len() as u64;

                        model.reset();
                        *chunks_since_decoder_reset = 0;
                        *consecutive_empty = 0;
                        replay_buffer.clear();

                        let mut replay_nonempty_chunks = 0i64;
                        let mut replay_inference_ms = 0i64;

                        for replay_chunk in &replay_chunks {
                            *chunks_since_decoder_reset += 1;
                            let infer_start = Instant::now();
                            match model.transcribe_chunk(replay_chunk) {
                                Ok(text) => {
                                    let infer_ms = infer_start.elapsed().as_millis() as i64;
                                    replay_inference_ms += infer_ms;
                                    *chunk_num += 1;
                                    let is_empty = text.is_empty();
                                    let preview = text_preview(&text, 200);
                                    let iteration_ms = iter_start.elapsed().as_millis() as i64;

                                    // Replay rows record vad_state as a literal
                                    // "speech" rather than the sampled state;
                                    // preserved as-is.
                                    diag.transcribe(&TranscribeRow {
                                        uptime_ms: ctx.session.uptime_ms(),
                                        chunk_num: *chunk_num,
                                        inference_ms: infer_ms,
                                        drain_samples: drain_count,
                                        drain_audio_ms,
                                        asr_buf_len: asr_buffer.len().saturating_sub(asr_consumed),
                                        is_empty,
                                        preview: &preview,
                                        vad_state: "speech",
                                        vad_ms,
                                        resample_ms,
                                        iteration_ms,
                                        chunk_source: "replay",
                                    });

                                    if !is_empty {
                                        replay_nonempty_chunks += 1;
                                        out.push(TranscriptUpdate {
                                            segment_id: ids.next(),
                                            is_final: true,
                                            text,
                                            speaker: None,
                                        });
                                    }
                                }
                                Err(e) => {
                                    diag.asr_error(&AsrErrorRow {
                                        uptime_ms: ctx.session.uptime_ms(),
                                        chunk_num: *chunk_num,
                                        inference_ms: infer_start.elapsed().as_millis() as i64,
                                        error_msg: &e.to_string(),
                                        vad_state: "speech",
                                        chunk_source: "replay",
                                    });
                                }
                            }
                        }

                        diag.mid_speech_reset(&MidSpeechResetRow {
                            uptime_ms: reset_uptime,
                            consecutive_empty: consecutive_empty_at_reset,
                            chunks_since_decoder_reset: chunks_at_reset,
                            replay_chunks: replay_chunk_count,
                            replay_nonempty_chunks,
                            replay_inference_ms,
                        });
                    }
                }
                Err(e) => {
                    diag.asr_error(&AsrErrorRow {
                        uptime_ms: ctx.session.uptime_ms(),
                        chunk_num: *chunk_num,
                        inference_ms: infer_start.elapsed().as_millis() as i64,
                        error_msg: &e.to_string(),
                        vad_state: vad_state_str,
                        chunk_source: "live",
                    });
                }
            }

            while pending_speech_end_resets
                .front()
                .is_some_and(|reset| asr_consumed >= reset.reset_after_samples)
            {
                let reset = pending_speech_end_resets
                    .pop_front()
                    .expect("front checked above");

                diag.speech_end(&SpeechEndRow {
                    uptime_ms: reset.uptime_ms,
                    speech_duration_ms: reset.speech_duration_ms,
                    consecutive_empty: *consecutive_empty,
                    chunks_since_decoder_reset: *chunks_since_decoder_reset,
                });

                *consecutive_empty = 0;
                model.reset();
                *chunks_since_decoder_reset = 0;
                replay_buffer.clear();
            }
        }
        if asr_consumed > 0 {
            asr_buffer.drain(..asr_consumed);
        }

        Ok(out)
    }
}

fn pad_to_chunk_boundary(buffer: &mut Vec<f32>, chunk_size: usize) {
    if buffer.is_empty() || chunk_size == 0 {
        return;
    }
    let remainder = buffer.len() % chunk_size;
    if remainder != 0 {
        buffer.resize(buffer.len() + chunk_size - remainder, 0.0);
    }
}

fn text_preview(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }

    let end = text
        .char_indices()
        .map(|(idx, _)| idx)
        .take_while(|&idx| idx <= max_bytes)
        .last()
        .unwrap_or(0);
    text[..end].to_string()
}

/// Check if text ends with sentence-ending punctuation (`.`, `?`, `!`),
/// filtering out ellipsis and decimal-looking patterns.
pub fn ends_with_sentence_punctuation(text: &str) -> bool {
    let trimmed = text.trim_end();
    if trimmed.is_empty() {
        return false;
    }
    match trimmed.as_bytes()[trimmed.len() - 1] {
        b'?' | b'!' => true,
        b'.' => {
            // Filter out ellipsis ("...")
            if trimmed.ends_with("...") {
                return false;
            }
            // Filter out decimal-looking patterns (digit before ".")
            let before_dot = &trimmed[..trimmed.len() - 1];
            let last_char = before_dot.trim_end().bytes().last();
            !matches!(last_char, Some(b'0'..=b'9'))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn period_is_sentence_punctuation() {
        assert!(ends_with_sentence_punctuation("Hello."));
    }

    #[test]
    fn question_mark_is_sentence_punctuation() {
        assert!(ends_with_sentence_punctuation("Hello?"));
    }

    #[test]
    fn exclamation_is_sentence_punctuation() {
        assert!(ends_with_sentence_punctuation("Hello!"));
    }

    #[test]
    fn ellipsis_is_not_sentence_punctuation() {
        assert!(!ends_with_sentence_punctuation("Hello..."));
    }

    #[test]
    fn digit_before_period_is_not_sentence_punctuation() {
        assert!(!ends_with_sentence_punctuation("3."));
        assert!(!ends_with_sentence_punctuation("The value is 3.14."));
    }

    #[test]
    fn word_before_period_is_sentence_punctuation() {
        assert!(ends_with_sentence_punctuation("end."));
        assert!(ends_with_sentence_punctuation("The end."));
    }

    #[test]
    fn empty_string_is_not_sentence_punctuation() {
        assert!(!ends_with_sentence_punctuation(""));
    }

    #[test]
    fn whitespace_only_is_not_sentence_punctuation() {
        assert!(!ends_with_sentence_punctuation("   "));
    }

    #[test]
    fn trailing_whitespace_is_trimmed() {
        assert!(ends_with_sentence_punctuation("Hello.  "));
        assert!(ends_with_sentence_punctuation("Hello?  "));
    }

    #[test]
    fn no_punctuation_is_not_sentence_ending() {
        assert!(!ends_with_sentence_punctuation("Hello"));
        assert!(!ends_with_sentence_punctuation("Hello,"));
        assert!(!ends_with_sentence_punctuation("Hello;"));
    }

    #[test]
    fn pad_to_chunk_boundary_pads_partial_chunk() {
        let mut buffer = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        pad_to_chunk_boundary(&mut buffer, 4);

        assert_eq!(buffer, vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn pad_to_chunk_boundary_preserves_aligned_buffer() {
        let mut buffer = vec![1.0, 2.0, 3.0, 4.0];

        pad_to_chunk_boundary(&mut buffer, 4);

        assert_eq!(buffer, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn pad_to_chunk_boundary_preserves_empty_buffer() {
        let mut buffer = Vec::new();

        pad_to_chunk_boundary(&mut buffer, 4);

        assert!(buffer.is_empty());
    }

    #[test]
    fn text_preview_does_not_split_utf8() {
        assert_eq!(text_preview("abcédef", 4), "abc");
        assert_eq!(text_preview("abcédef", 5), "abcé");
    }

    #[test]
    fn full_text_punctuation_can_differ_from_preview() {
        let long_text = format!("{}.", "word ".repeat(60));
        let preview = text_preview(&long_text, 200);

        assert!(!ends_with_sentence_punctuation(&preview));
        assert!(ends_with_sentence_punctuation(&long_text));
    }

    #[test]
    fn replay_buffer_keeps_latest_chunks() {
        let mut buffer = ReplayBuffer::new(2);

        buffer.push(&[1.0]);
        buffer.push(&[2.0]);
        buffer.push(&[3.0]);

        assert_eq!(buffer.snapshot(), vec![vec![2.0], vec![3.0]]);
    }

    #[test]
    fn replay_buffer_capacity_update_truncates_old_chunks() {
        let mut buffer = ReplayBuffer::new(3);

        buffer.push(&[1.0]);
        buffer.push(&[2.0]);
        buffer.push(&[3.0]);
        buffer.update_capacity(1);

        assert_eq!(buffer.snapshot(), vec![vec![3.0]]);
    }

    #[test]
    fn replay_buffer_zero_capacity_keeps_no_chunks() {
        let mut buffer = ReplayBuffer::new(0);

        buffer.push(&[1.0]);

        assert!(buffer.snapshot().is_empty());
    }

    #[test]
    fn capabilities_are_append_only_and_vad_gated() {
        let caps = NemotronBackend::new().capabilities();
        assert!(!caps.revises_output);
        assert!(!caps.diarization);
        assert_eq!(caps.audio_gating, AudioGating::VadGated);
        assert_eq!(caps.activity_source, ActivitySource::Vad);
    }
}
