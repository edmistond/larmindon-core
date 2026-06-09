//! Pluggable speech-recognition engine abstraction.
//!
//! The processing loop in `audio_engine.rs` owns the shared pipeline stages
//! (capture drain, resample, AGC, VAD) and pushes speech-gated audio into a
//! [`SpeechEngine`]. Engine implementations live in sibling crates
//! (e.g. `larmindon-engine-nemotron`) and are registered with the
//! [`registry::EngineRegistry`] by the application shell.

pub mod registry;

use crate::diagnostics::DiagSink;

/// One transcription segment update from an engine.
///
/// Engines that revise hypotheses emit several updates with the same
/// `segment_id` and `is_final: false`, each replacing the previous text, then
/// one closing update with `is_final: true`. Engines whose output is
/// append-only (e.g. Nemotron) emit every segment already finalized.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SegmentUpdate {
    /// Engine-local id, unique within one engine instance. The core remaps it
    /// to a globally unique id before emitting to the UI.
    pub segment_id: u64,
    pub text: String,
    pub is_final: bool,
}

#[derive(Debug)]
pub enum EngineError {
    /// The session cannot continue (model failed to load, backend died).
    Fatal(String),
    /// A recoverable hiccup (one bad chunk); log and continue.
    Transient(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Fatal(msg) => write!(f, "fatal engine error: {}", msg),
            EngineError::Transient(msg) => write!(f, "transient engine error: {}", msg),
        }
    }
}

impl std::error::Error for EngineError {}

/// Per-session resources handed to the engine at `begin_session`.
#[derive(Clone)]
pub struct SessionContext {
    pub diag: DiagSink,
}

/// A speech-recognition backend.
///
/// Lifecycle: constructed once by its factory (cheap — no model load), then
/// `begin_session`/`end_session` per transcription session. The engine
/// instance survives across sessions so heavyweight models stay loaded; the
/// per-session config is re-applied on every `begin_session`.
///
/// All methods are called on the processing thread. Audio arrives as 16 kHz
/// mono f32 (post-AGC), gated by VAD: `feed` is only called between
/// `on_speech_start` and `on_speech_end`, plus once for the pre-speech ring
/// buffer right after `on_speech_start`.
pub trait SpeechEngine: Send {
    fn engine_id(&self) -> &'static str;

    /// Start a session. `config` is this engine's config blob for the session
    /// (may differ from the one the engine was created with — e.g. a changed
    /// chunk size — but fields covered by the factory's `cache_key` are
    /// guaranteed unchanged for a reused instance). Loads the model if not
    /// already resident.
    fn begin_session(
        &mut self,
        ctx: SessionContext,
        config: &serde_json::Value,
    ) -> Result<(), EngineError>;

    /// VAD opened a speech segment. The pre-speech ring buffer arrives via the
    /// next `feed` call.
    fn on_speech_start(&mut self);

    /// Speech-gated audio at any granularity.
    fn feed(&mut self, samples: &[f32]) -> Result<Vec<SegmentUpdate>, EngineError>;

    /// VAD closed the speech segment. Chunk-based engines pad and drain their
    /// internal buffer; streaming engines flush so in-flight hypotheses
    /// finalize (possibly asynchronously, via a later `poll`).
    fn on_speech_end(&mut self) -> Result<Vec<SegmentUpdate>, EngineError>;

    /// Non-blocking drain of asynchronously produced results. Called every
    /// loop iteration, including idle ones, so callback- or socket-driven
    /// engines surface results during silence. Sync engines use the default.
    fn poll(&mut self) -> Result<Vec<SegmentUpdate>, EngineError> {
        Ok(Vec::new())
    }

    /// Hot-reload of this engine's config blob mid-session. Engines apply the
    /// subset of fields that can change live and ignore the rest.
    fn update_config(&mut self, config: &serde_json::Value);

    /// Tear down session state, returning any final updates for in-flight
    /// segments. Keeps the heavyweight model resident for the next session.
    fn end_session(&mut self) -> Result<Vec<SegmentUpdate>, EngineError>;
}
