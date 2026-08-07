//! Speech-recognition backends.
//!
//! A backend declares what it can do via [`AsrCapabilities`]; the processing
//! loop adapts its audio routing and its meter signal accordingly. Adding a
//! provider means adding a module and one arm to [`create_backend`] — there is
//! deliberately no registry, no factory trait and no config descriptors.
//!
//! The audio path is split in two on purpose:
//!
//! * [`AsrBackend::push_audio`] runs inside the VAD's 512-sample frame loop and
//!   may only buffer.
//! * [`AsrBackend::process`] runs exactly once per loop iteration and is the
//!   only place recognition happens.
//!
//! Collapsing these into a single `feed()` would move chunk-based recognition
//! inside the frame loop, letting the sampled VAD state flip part-way through
//! what is currently one atomic pass. That state gates the decoder-reset
//! heuristics, so it would change transcripts.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::diag::DiagSink;
use crate::settings::Settings;
use crate::vad::VadState;

pub mod nemotron;
#[cfg(feature = "asr-soniox")]
pub mod soniox;

/// A stable speaker identity within one session. Serializes as a bare JSON
/// string, so the frontend sees `speaker: "1" | null`. Providers without
/// diarization never produce one.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SpeakerId(pub String);

/// One segment of transcript, created or revised.
///
/// Contract every backend must honour, because the frontend store depends on it:
///
/// * `text` is the COMPLETE text of the segment, never a delta.
/// * A `segment_id` may be emitted any number of times with `is_final: false`,
///   then AT MOST ONCE with `is_final: true`, and never again after that.
/// * Ids increase monotonically in emission order.
///
/// An append-only backend emits every segment already final, each id once.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscriptUpdate {
    pub segment_id: u64,
    pub is_final: bool,
    pub text: String,
    /// `None` when the provider does not diarize, or has not yet attributed
    /// this segment. The UI must not invent a label for `None`.
    pub speaker: Option<SpeakerId>,
}

#[derive(Debug)]
pub enum AsrError {
    /// The session cannot continue (model load failed, auth rejected, socket
    /// dead past retry). Ends the session and surfaces to the UI.
    Fatal(String),
    /// Recoverable hiccup. Logged to diagnostics and surfaced as a status
    /// notice; the session continues.
    Transient(String),
}

impl std::fmt::Display for AsrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsrError::Fatal(m) => write!(f, "{m}"),
            AsrError::Transient(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for AsrError {}

/// How the pipeline routes audio into a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioGating {
    /// Speech only: the pre-speech ring buffer at onset, then each speech
    /// frame. Cheaper, and correct for chunk-based local models.
    VadGated,
    /// Every frame, silence included, exactly once. Required by providers that
    /// do their own semantic endpointing and diarization.
    ///
    /// The pre-speech ring buffer is never delivered in this mode — those
    /// samples already went out as ordinary frames, so replaying them would
    /// duplicate about 500 ms of audio at every speech onset.
    Continuous,
}

/// What drives the "hot" (coloured) state of the level meter.
///
/// That state is load-bearing UI: it means "something is actually hearing
/// speech right now". For a VAD-gated local model the VAD is exactly that
/// signal. For a remote model the VAD gates nothing, so the honest signal is
/// "the server is returning speech tokens".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivitySource {
    /// Use the pipeline's Silero VAD state.
    Vad,
    /// Use [`AsrBackend::last_speech_at`], held for `hold` past the last
    /// observation.
    ///
    /// `hold` must exceed the provider's idle message cadence. A provider that
    /// answers roughly once a second even in silence will make the indicator
    /// strobe if `hold` is shorter than that, and will make it lie about
    /// silence if `hold` is much longer.
    Backend { hold: Duration },
}

pub struct AsrCapabilities {
    /// Emits non-final segments that later change.
    pub revises_output: bool,
    /// Populates [`TranscriptUpdate::speaker`].
    pub diarization: bool,
    pub audio_gating: AudioGating,
    pub activity_source: ActivitySource,
}

/// Monotonic segment ids, owned by the engine and shared across sessions and
/// provider switches so an id is never reused while a stale segment carrying it
/// could still be on screen.
#[derive(Clone, Default)]
pub struct SegmentIds(Arc<AtomicU64>);

impl SegmentIds {
    pub fn new() -> Self {
        Self(Arc::new(AtomicU64::new(1)))
    }

    pub fn next(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

/// Borrowed for the duration of a single call, never stored by a backend.
///
/// That restriction is what keeps every backend `Send` — so it can still be
/// handed back through the processing thread's `JoinHandle` for cross-session
/// caching — while the SQLite `Connection` inside [`DiagSink`] stays
/// single-owner on the processing thread, with no `Arc` and no `Mutex`.
pub struct SessionContext<'a> {
    /// The settings the session started with. Hot-reloaded values arrive via
    /// [`AsrBackend::update_settings`]; this does not change mid-session.
    pub settings: &'a Settings,
    pub diag: &'a DiagSink,
    pub ids: &'a SegmentIds,
    pub loop_start: Instant,
    pub sample_rate: usize,
    pub input_rate: usize,
    pub needs_resample: bool,
}

impl SessionContext<'_> {
    pub fn uptime_ms(&self) -> i64 {
        self.loop_start.elapsed().as_millis() as i64
    }
}

/// Per-iteration pipeline facts the ASR stage records, mirroring field for field
/// the values the loop writes into the `events` table.
#[derive(Debug, Clone, Copy)]
pub struct IterationStats {
    pub iter_start: Instant,
    pub drain_samples: usize,
    pub drain_audio_ms: f64,
    pub resample_ms: i64,
    pub vad_ms: i64,
    /// Sampled once, after the VAD frame loop. The VAD is not advanced again
    /// during the ASR stage, so this equals what a live `vad.state()` call
    /// inside it would return.
    pub vad_state: VadState,
}

pub struct AsrContext<'a> {
    pub session: &'a SessionContext<'a>,
    pub stats: &'a IterationStats,
}

pub trait AsrBackend: Send {
    fn id(&self) -> &'static str;
    fn capabilities(&self) -> AsrCapabilities;

    /// Whether the engine may keep this instance alive between sessions.
    fn is_cacheable(&self) -> bool {
        false
    }

    /// Load models / open connections. Runs on the processing thread.
    fn begin_session(&mut self, ctx: &SessionContext) -> Result<(), AsrError>;

    // ---- VAD stage --------------------------------------------------------
    // These run inside the 512-sample frame loop. They take no context, write
    // no diagnostics and return no updates, by design.

    /// 16 kHz mono f32, post-AGC, in capture order. Buffer only.
    fn push_audio(&mut self, samples: &[f32]);

    /// The VAD opened a speech segment. Informational for continuous backends.
    fn mark_speech_start(&mut self) {}

    /// The VAD closed a speech segment. Informational for continuous backends.
    fn mark_speech_end(&mut self, uptime_ms: i64, speech_duration_ms: f64) {
        let _ = (uptime_ms, speech_duration_ms);
    }

    // ---- ASR stage: exactly once per loop iteration ------------------------

    fn process(&mut self, ctx: &AsrContext) -> Result<Vec<TranscriptUpdate>, AsrError>;

    /// Called on iterations where the capture buffer drained empty, so
    /// socket-driven backends still surface results while audio is starved.
    /// The no-op default is what leaves synchronous backends unchanged.
    fn poll(&mut self, ctx: &SessionContext) -> Result<Vec<TranscriptUpdate>, AsrError> {
        let _ = ctx;
        Ok(Vec::new())
    }

    /// Instant of the last evidence of SPEECH — not of traffic, and not of a
    /// message merely arriving. Only consulted when `activity_source` is
    /// [`ActivitySource::Backend`].
    fn last_speech_at(&self) -> Option<Instant> {
        None
    }

    /// Chunks processed so far, for the `shutdown` diagnostics row.
    fn chunks_processed(&self) -> u64 {
        0
    }

    /// Hot-reload of settings mid-session. Apply what can change live, ignore
    /// the rest.
    fn update_settings(&mut self, settings: &Settings);

    /// Tear down, returning final updates for any open segments.
    fn end_session(&mut self, ctx: &SessionContext) -> Result<Vec<TranscriptUpdate>, AsrError>;
}

/// Builds the backend named by `settings.asr_provider`, reusing `cached` when
/// it is the same cacheable provider.
pub fn create_backend(
    settings: &Settings,
    cached: Option<Box<dyn AsrBackend>>,
) -> Result<Box<dyn AsrBackend>, AsrError> {
    let reusable = cached.filter(|b| b.is_cacheable() && b.id() == settings.asr_provider.as_str());

    match settings.asr_provider.as_str() {
        "nemotron" => Ok(reusable.unwrap_or_else(|| Box::new(nemotron::NemotronBackend::new()))),
        #[cfg(feature = "asr-soniox")]
        "soniox" => Ok(Box::new(soniox::SonioxBackend::new())),
        other => Err(AsrError::Fatal(format!(
            "Unknown transcription provider '{other}'"
        ))),
    }
}
