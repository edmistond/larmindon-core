//! Nemotron (parakeet-rs) speech engine.
//!
//! Nemotron is a streaming, append-only model: it consumes fixed-size audio
//! chunks and every result it emits is final — it never revises earlier
//! output. This crate adapts that model to the push-based
//! [`SpeechEngine`] interface by buffering fed audio into chunks, and carries
//! the Nemotron-specific decoder-reset heuristics:
//!
//! - **Punctuation reset**: reset the decoder after sentence-ending
//!   punctuation for a clean slate at sentence boundaries.
//! - **Mid-speech (stuck decoder) reset**: if the model emits N consecutive
//!   empty results during VAD-detected speech, reset and replay the last N
//!   chunks once to recover from stuck decoder states.
//! - **Speech-end reset**: pad to a chunk boundary, drain, and reset when VAD
//!   closes a speech segment.

use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::time::Instant;

#[cfg(any(feature = "webgpu", feature = "directml", feature = "migraphx"))]
use parakeet_rs::ExecutionProvider;
use parakeet_rs::{ExecutionConfig, Nemotron};
use serde::{Deserialize, Serialize};

use larmindon_core::diagnostics::{text_preview, DiagSink};
use larmindon_core::engine::registry::{
    ConfigField, EngineDescriptor, EngineFactory, EngineKind, EnumOption, FieldType,
};
use larmindon_core::engine::{EngineError, SegmentUpdate, SessionContext, SpeechEngine};
use larmindon_core::settings::{chunk_ms_to_samples, expand_tilde};

pub const ENGINE_ID: &str = "nemotron";
const VALID_CHUNK_MS: &[usize] = &[80, 160, 560, 1120];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NemotronConfig {
    pub model_path: String,
    pub chunk_ms: usize,
    pub intra_threads: usize,
    pub inter_threads: usize,
    pub punctuation_reset: bool,
    pub empty_reset_threshold: u32,
}

impl Default for NemotronConfig {
    fn default() -> Self {
        Self {
            model_path: "~/projects/prs-nemotron/".to_string(),
            chunk_ms: 560,
            intra_threads: 2,
            inter_threads: 1,
            punctuation_reset: true,
            empty_reset_threshold: 6,
        }
    }
}

impl NemotronConfig {
    fn parse(config: &serde_json::Value) -> Result<Self, String> {
        let cfg: Self = serde_json::from_value(config.clone())
            .map_err(|e| format!("Invalid nemotron config: {}", e))?;
        if !VALID_CHUNK_MS.contains(&cfg.chunk_ms) {
            return Err(format!(
                "Invalid chunk_ms {}; must be one of {:?}",
                cfg.chunk_ms, VALID_CHUNK_MS
            ));
        }
        if cfg.intra_threads < 1 {
            return Err("intra_threads must be at least 1".to_string());
        }
        if cfg.inter_threads < 1 {
            return Err("inter_threads must be at least 1".to_string());
        }
        if cfg.empty_reset_threshold < 1 {
            return Err("empty_reset_threshold must be at least 1".to_string());
        }
        if cfg.model_path.trim().is_empty() {
            return Err("model_path cannot be empty".to_string());
        }
        Ok(cfg)
    }
}

pub struct NemotronFactory;

impl EngineFactory for NemotronFactory {
    fn descriptor(&self) -> EngineDescriptor {
        let defaults = NemotronConfig::default();
        EngineDescriptor {
            id: ENGINE_ID,
            name: "Nemotron (parakeet-rs)",
            kind: EngineKind::Local,
            emits_partials: false,
            config_fields: vec![
                ConfigField {
                    key: "model_path",
                    label: "Model Path",
                    field: FieldType::Path { directory: true },
                    default: defaults.model_path.clone().into(),
                    env_var: None,
                    help: Some("Directory containing the Nemotron streaming model files"),
                },
                ConfigField {
                    key: "chunk_ms",
                    label: "Chunk Size",
                    field: FieldType::Enum {
                        options: VALID_CHUNK_MS
                            .iter()
                            .map(|&ms| EnumOption {
                                value: ms.into(),
                                label: format!("{} ms", ms),
                            })
                            .collect(),
                    },
                    default: defaults.chunk_ms.into(),
                    env_var: Some("CHUNK_MS"),
                    help: Some("Audio chunk duration fed to the model"),
                },
                ConfigField {
                    key: "intra_threads",
                    label: "Intra-op Threads",
                    field: FieldType::Int { min: 1, max: 32 },
                    default: defaults.intra_threads.into(),
                    env_var: Some("INTRA_THREADS"),
                    help: Some("ONNX intra-op parallelism"),
                },
                ConfigField {
                    key: "inter_threads",
                    label: "Inter-op Threads",
                    field: FieldType::Int { min: 1, max: 32 },
                    default: defaults.inter_threads.into(),
                    env_var: Some("INTER_THREADS"),
                    help: Some("ONNX inter-op parallelism"),
                },
                ConfigField {
                    key: "punctuation_reset",
                    label: "Punctuation-based decoder reset",
                    field: FieldType::Bool,
                    default: defaults.punctuation_reset.into(),
                    env_var: Some("PUNCTUATION_RESET"),
                    help: Some("Reset the decoder after sentence-ending punctuation"),
                },
                ConfigField {
                    key: "empty_reset_threshold",
                    label: "Empty chunk reset threshold",
                    field: FieldType::Int { min: 1, max: 50 },
                    default: defaults.empty_reset_threshold.into(),
                    env_var: None,
                    help: Some(
                        "Consecutive empty chunks during speech before resetting the decoder \
                         and replaying buffered audio",
                    ),
                },
            ],
        }
    }

    fn default_config(&self) -> serde_json::Value {
        serde_json::to_value(NemotronConfig::default()).expect("config serializes")
    }

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), String> {
        NemotronConfig::parse(config).map(|_| ())
    }

    fn cache_key(&self, config: &serde_json::Value) -> u64 {
        // Only fields that force a model reload. chunk_ms and the reset knobs
        // apply to a reused instance at begin_session / hot-reload.
        let cfg = NemotronConfig::parse(config).unwrap_or_default();
        let mut hasher = DefaultHasher::new();
        expand_tilde(&cfg.model_path)
            .to_string_lossy()
            .hash(&mut hasher);
        cfg.intra_threads.hash(&mut hasher);
        cfg.inter_threads.hash(&mut hasher);
        hasher.finish()
    }

    fn create(&self, config: &serde_json::Value) -> Result<Box<dyn SpeechEngine>, EngineError> {
        let cfg = NemotronConfig::parse(config).map_err(EngineError::Fatal)?;
        Ok(Box::new(NemotronEngine::new(cfg)))
    }
}

struct ReplayBuffer {
    chunks: VecDeque<Vec<f32>>,
    max_chunks: usize,
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

pub struct NemotronEngine {
    config: NemotronConfig,
    /// Loaded lazily in `begin_session`; survives `end_session` so the next
    /// session skips the (slow) model load.
    model: Option<Nemotron>,
    chunk_size: usize,
    asr_buffer: Vec<f32>,
    replay_buffer: ReplayBuffer,
    /// Whether the audio being fed belongs to an open VAD speech segment.
    /// Chunks drained after speech end (padding tail) don't count toward the
    /// stuck-decoder heuristic and aren't eligible for replay.
    in_speech: bool,
    consecutive_empty: u32,
    chunks_since_decoder_reset: u64,
    chunk_num: u64,
    next_segment_id: u64,
    diag: DiagSink,
}

impl NemotronEngine {
    fn new(config: NemotronConfig) -> Self {
        let chunk_size = chunk_ms_to_samples(config.chunk_ms);
        let replay_capacity = config.empty_reset_threshold as usize;
        Self {
            config,
            model: None,
            chunk_size,
            asr_buffer: Vec::new(),
            replay_buffer: ReplayBuffer::new(replay_capacity),
            in_speech: false,
            consecutive_empty: 0,
            chunks_since_decoder_reset: 0,
            chunk_num: 0,
            next_segment_id: 0,
            diag: DiagSink::disabled(),
        }
    }

    fn final_segment(&mut self, text: String) -> SegmentUpdate {
        let segment_id = self.next_segment_id;
        self.next_segment_id += 1;
        SegmentUpdate {
            segment_id,
            text,
            is_final: true,
        }
    }

    fn reset_decoder(&mut self) {
        if let Some(model) = self.model.as_mut() {
            model.reset();
        }
        self.chunks_since_decoder_reset = 0;
        self.consecutive_empty = 0;
        self.replay_buffer.clear();
    }

    fn log_transcribe_event(&self, infer_ms: i64, text: &str, source: &str) {
        let chunk_num = self.chunk_num;
        let asr_buf_len = self.asr_buffer.len();
        let vad_state = if self.in_speech { "speech" } else { "silence" };
        let is_empty = text.is_empty();
        let preview = text_preview(text, 200);
        self.diag.with_conn(|conn, session_id, uptime| {
            let _ = conn.execute(
                "INSERT INTO events (session_id, uptime_ms, event_type, chunk_num,
                 inference_ms, asr_buf_len, text_empty, text_preview, vad_state, chunk_source)
                 VALUES (?1, ?2, 'transcribe', ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    session_id,
                    uptime,
                    chunk_num as i64,
                    infer_ms,
                    asr_buf_len as i64,
                    is_empty as i64,
                    preview,
                    vad_state,
                    source,
                ],
            );
        });
    }

    fn log_asr_error(&self, infer_ms: i64, error: &str, source: &str) {
        let chunk_num = self.chunk_num;
        let vad_state = if self.in_speech { "speech" } else { "silence" };
        self.diag.with_conn(|conn, session_id, uptime| {
            let _ = conn.execute(
                "INSERT INTO events (session_id, uptime_ms, event_type, chunk_num,
                 inference_ms, error_msg, vad_state, chunk_source)
                 VALUES (?1, ?2, 'asr_error', ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    session_id,
                    uptime,
                    chunk_num as i64,
                    infer_ms,
                    error,
                    vad_state,
                    source,
                ],
            );
        });
    }

    /// Consume complete chunks from the internal buffer, running inference and
    /// the Nemotron reset heuristics. Returns finalized segment updates.
    fn drain_chunks(&mut self) -> Vec<SegmentUpdate> {
        let mut updates = Vec::new();
        let mut consumed = 0;

        while self.asr_buffer.len().saturating_sub(consumed) >= self.chunk_size {
            let chunk: Vec<f32> = self.asr_buffer[consumed..consumed + self.chunk_size].to_vec();
            consumed += self.chunk_size;

            if self.in_speech {
                self.replay_buffer.push(&chunk);
            }
            self.chunks_since_decoder_reset += 1;

            let Some(model) = self.model.as_mut() else {
                break;
            };
            let infer_start = Instant::now();
            match model.transcribe_chunk(&chunk) {
                Ok(text) => {
                    let infer_ms = infer_start.elapsed().as_millis() as i64;
                    self.chunk_num += 1;
                    let is_empty = text.is_empty();

                    if is_empty && self.in_speech {
                        self.consecutive_empty += 1;
                    } else {
                        self.consecutive_empty = 0;
                    }

                    self.log_transcribe_event(infer_ms, &text, "live");

                    // Punctuation-based decoder reset
                    if self.config.punctuation_reset
                        && !is_empty
                        && ends_with_sentence_punctuation(&text)
                    {
                        let consecutive_empty = self.consecutive_empty;
                        let chunks_at_reset = self.chunks_since_decoder_reset;
                        let audio_ms_at_reset =
                            chunks_to_audio_ms(chunks_at_reset, self.config.chunk_ms);
                        self.diag.with_conn(|conn, session_id, uptime| {
                            let _ = conn.execute(
                                "INSERT INTO vad_events (
                                    session_id, uptime_ms, event_type, consecutive_empty,
                                    chunks_since_decoder_reset, audio_ms_since_decoder_reset
                                 )
                                 VALUES (?1, ?2, 'punctuation_reset', ?3, ?4, ?5)",
                                rusqlite::params![
                                    session_id,
                                    uptime,
                                    consecutive_empty as i64,
                                    chunks_at_reset as i64,
                                    audio_ms_at_reset,
                                ],
                            );
                        });
                        self.reset_decoder();
                    }

                    if !is_empty {
                        updates.push(self.final_segment(text));
                    }

                    // Mid-speech (stuck decoder) reset heuristic
                    if self.consecutive_empty >= self.config.empty_reset_threshold && self.in_speech
                    {
                        updates.extend(self.run_mid_speech_reset());
                    }
                }
                Err(e) => {
                    let infer_ms = infer_start.elapsed().as_millis() as i64;
                    self.log_asr_error(infer_ms, &e.to_string(), "live");
                }
            }
        }

        if consumed > 0 {
            self.asr_buffer.drain(..consumed);
        }
        updates
    }

    /// Reset the decoder and replay the buffered speech chunks once,
    /// re-emitting any text they produce.
    fn run_mid_speech_reset(&mut self) -> Vec<SegmentUpdate> {
        let consecutive_empty_at_reset = self.consecutive_empty;
        let chunks_at_reset = self.chunks_since_decoder_reset;
        let audio_ms_at_reset = chunks_to_audio_ms(chunks_at_reset, self.config.chunk_ms);
        let replay_chunks = self.replay_buffer.snapshot();
        let replay_chunk_count = replay_chunks.len() as i64;
        let replay_audio_ms = chunks_to_audio_ms(replay_chunks.len() as u64, self.config.chunk_ms);

        self.reset_decoder();

        let mut updates = Vec::new();
        let mut replay_nonempty_chunks = 0i64;
        let mut replay_inference_ms = 0i64;

        for replay_chunk in &replay_chunks {
            self.chunks_since_decoder_reset += 1;
            let Some(model) = self.model.as_mut() else {
                break;
            };
            let infer_start = Instant::now();
            match model.transcribe_chunk(replay_chunk) {
                Ok(text) => {
                    let infer_ms = infer_start.elapsed().as_millis() as i64;
                    replay_inference_ms += infer_ms;
                    self.chunk_num += 1;
                    self.log_transcribe_event(infer_ms, &text, "replay");
                    if !text.is_empty() {
                        replay_nonempty_chunks += 1;
                        updates.push(self.final_segment(text));
                    }
                }
                Err(e) => {
                    let infer_ms = infer_start.elapsed().as_millis() as i64;
                    self.log_asr_error(infer_ms, &e.to_string(), "replay");
                }
            }
        }

        self.diag.with_conn(|conn, session_id, uptime| {
            let _ = conn.execute(
                "INSERT INTO vad_events (
                    session_id, uptime_ms, event_type, consecutive_empty,
                    chunks_since_decoder_reset, audio_ms_since_decoder_reset,
                    replay_chunks, replay_audio_ms, replay_nonempty_chunks,
                    replay_inference_ms
                 )
                 VALUES (?1, ?2, 'mid_speech_reset', ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    session_id,
                    uptime,
                    consecutive_empty_at_reset as i64,
                    chunks_at_reset as i64,
                    audio_ms_at_reset,
                    replay_chunk_count,
                    replay_audio_ms,
                    replay_nonempty_chunks,
                    replay_inference_ms,
                ],
            );
        });

        updates
    }
}

impl SpeechEngine for NemotronEngine {
    fn engine_id(&self) -> &'static str {
        ENGINE_ID
    }

    fn begin_session(
        &mut self,
        ctx: SessionContext,
        config: &serde_json::Value,
    ) -> Result<(), EngineError> {
        // Re-apply the full config: a reused instance may carry a stale
        // chunk_ms or reset knobs. cache_key guarantees model_path/threads
        // didn't change for a reused instance.
        let cfg = NemotronConfig::parse(config).map_err(EngineError::Fatal)?;
        self.chunk_size = chunk_ms_to_samples(cfg.chunk_ms);
        self.replay_buffer = ReplayBuffer::new(cfg.empty_reset_threshold as usize);
        self.config = cfg;

        self.diag = ctx.diag;
        self.asr_buffer.clear();
        self.in_speech = false;
        self.consecutive_empty = 0;
        self.chunks_since_decoder_reset = 0;
        self.chunk_num = 0;

        println!(
            "Nemotron session: chunk_ms={}ms ({} samples), intra={}, inter={}, punctuation_reset={}, empty_reset_threshold={}",
            self.config.chunk_ms, self.chunk_size, self.config.intra_threads,
            self.config.inter_threads, self.config.punctuation_reset,
            self.config.empty_reset_threshold
        );

        if let Some(model) = self.model.as_mut() {
            println!("Using cached Nemotron model (skipping reload)");
            model.reset();
        } else {
            let model_path = expand_tilde(&self.config.model_path);
            println!(
                "Loading Nemotron model from {} (intra_threads={}, inter_threads={})...",
                model_path.display(),
                self.config.intra_threads,
                self.config.inter_threads
            );
            #[allow(unused_mut)]
            let mut model_config = ExecutionConfig::new()
                .with_intra_threads(self.config.intra_threads)
                .with_inter_threads(self.config.inter_threads);

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

            let model = Nemotron::from_pretrained(&model_path, Some(model_config))
                .map_err(|e| EngineError::Fatal(e.to_string()))?;
            println!("Model loaded.");
            self.model = Some(model);
        }

        self.diag.set_session_chunk_size(self.chunk_size);
        Ok(())
    }

    fn on_speech_start(&mut self) {
        self.in_speech = true;
        self.consecutive_empty = 0;
        self.replay_buffer.clear();
    }

    fn feed(&mut self, samples: &[f32]) -> Result<Vec<SegmentUpdate>, EngineError> {
        self.asr_buffer.extend_from_slice(samples);
        Ok(self.drain_chunks())
    }

    fn on_speech_end(&mut self) -> Result<Vec<SegmentUpdate>, EngineError> {
        // The padded tail isn't "speech" for the stuck-decoder heuristic:
        // empties here are expected and must not trigger a replay.
        self.in_speech = false;
        pad_to_chunk_boundary(&mut self.asr_buffer, self.chunk_size);
        let updates = self.drain_chunks();
        self.reset_decoder();
        Ok(updates)
    }

    fn update_config(&mut self, config: &serde_json::Value) {
        match NemotronConfig::parse(config) {
            Ok(cfg) => {
                if cfg.punctuation_reset != self.config.punctuation_reset
                    || cfg.empty_reset_threshold != self.config.empty_reset_threshold
                {
                    println!(
                        "[diag] Nemotron hot-reload: punctuation_reset={}, empty_reset_threshold={}",
                        cfg.punctuation_reset, cfg.empty_reset_threshold
                    );
                }
                self.config.punctuation_reset = cfg.punctuation_reset;
                self.config.empty_reset_threshold = cfg.empty_reset_threshold;
                self.replay_buffer
                    .update_capacity(cfg.empty_reset_threshold as usize);
                // model_path/threads/chunk_ms take effect on the next session.
            }
            Err(e) => eprintln!("[diag] Ignoring invalid nemotron config update: {}", e),
        }
    }

    fn end_session(&mut self) -> Result<Vec<SegmentUpdate>, EngineError> {
        // Sub-chunk remainder is dropped (matches pre-refactor behavior); the
        // model stays loaded for the next session.
        self.asr_buffer.clear();
        self.replay_buffer.clear();
        self.in_speech = false;
        self.diag = DiagSink::disabled();
        Ok(Vec::new())
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

fn chunks_to_audio_ms(chunks: u64, chunk_ms: usize) -> i64 {
    chunks.saturating_mul(chunk_ms as u64).min(i64::MAX as u64) as i64
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
    use larmindon_core::diagnostics::text_preview;

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
    fn full_text_punctuation_can_differ_from_preview() {
        let long_text = format!("{}.", "word ".repeat(60));
        let preview = text_preview(&long_text, 200);

        assert!(!ends_with_sentence_punctuation(&preview));
        assert!(ends_with_sentence_punctuation(&long_text));
    }

    #[test]
    fn chunks_to_audio_ms_uses_chunk_duration() {
        assert_eq!(chunks_to_audio_ms(6, 560), 3360);
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
    fn default_config_is_valid() {
        let factory = NemotronFactory;
        assert!(factory.validate_config(&factory.default_config()).is_ok());
    }

    #[test]
    fn cache_key_ignores_hot_reloadable_fields() {
        let factory = NemotronFactory;
        let a = factory.default_config();
        let mut b = a.clone();
        b["chunk_ms"] = 160.into();
        b["punctuation_reset"] = false.into();
        b["empty_reset_threshold"] = 12.into();
        assert_eq!(factory.cache_key(&a), factory.cache_key(&b));

        let mut c = a.clone();
        c["intra_threads"] = 4.into();
        assert_ne!(factory.cache_key(&a), factory.cache_key(&c));
    }
}
