pub mod agc;
pub mod asr;
pub mod audio_capture;
pub mod audio_engine;
pub mod diag;
pub mod settings;
pub mod vad;

use asr::TranscriptUpdate;
use audio_capture::AudioDevice;

/// Severity of a non-fatal engine notice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum StatusLevel {
    Info,
    Warn,
}

/// Trait for receiving events from the audio engine.
///
/// Implementations bridge the core engine to a specific UI framework.
/// For example, a Tauri app emits events via `AppHandle`, a GTK app sends
/// via `mpsc::Sender<UiEvent>`, and tests collect into a `Vec`.
pub trait EngineEventSink: Send + Clone + 'static {
    /// Called when a backend creates or revises a transcript segment.
    ///
    /// A segment may be revised repeatedly while `is_final` is false, then
    /// finalized once. `text` is always the complete text of the segment.
    fn on_transcript_update(&self, update: TranscriptUpdate);

    /// Called when an error occurs during transcription.
    fn on_error(&self, message: String);

    /// Called when the audio source is switched to a different device.
    fn on_source_switched(&self, device_id: String);

    /// Called when the available device list changes (PipeWire watcher).
    fn on_devices_changed(&self, devices: Vec<AudioDevice>);

    /// Called at a UI-friendly cadence with a normalized audio level and
    /// whether speech is currently being heard. Sinks that do not display
    /// metering can ignore it.
    ///
    /// The activity flag is not always the VAD: a backend that does its own
    /// endpointing reports it from its own evidence of speech (see
    /// `asr::ActivitySource`). It means "something is hearing speech right
    /// now" regardless of which component decided that.
    fn on_audio_level(&self, _level: f32, _speech_active: bool) {}

    /// Non-fatal engine notices (reconnecting, degraded, recovered).
    /// Defaulted so existing sinks stay source-compatible.
    fn on_status(&self, _level: StatusLevel, _message: String) {}
}
