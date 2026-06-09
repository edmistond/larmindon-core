pub mod agc;
pub mod audio_capture;
pub mod audio_engine;
pub mod diagnostics;
pub mod engine;
pub mod settings;
pub mod vad;

use audio_capture::AudioDevice;
use engine::SegmentUpdate;

/// Trait for receiving events from the audio engine.
///
/// Implementations bridge the core engine to a specific UI framework.
/// For example, a Tauri app emits events via `AppHandle`, a GTK app sends
/// via `mpsc::Sender<UiEvent>`, and tests collect into a `Vec`.
pub trait EngineEventSink: Send + Clone + 'static {
    /// Called when the active engine produces or revises a transcription
    /// segment. `update.segment_id` is globally unique; a transient segment
    /// (`is_final: false`) is replaced wholesale by later updates with the
    /// same id until one arrives with `is_final: true`.
    fn on_segment_update(&self, update: SegmentUpdate);

    /// Called when an error occurs during transcription.
    fn on_error(&self, message: String);

    /// Called when the audio source is switched to a different device.
    fn on_source_switched(&self, device_id: String);

    /// Called when the available device list changes (PipeWire watcher).
    fn on_devices_changed(&self, devices: Vec<AudioDevice>);
}
