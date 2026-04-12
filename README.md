# larmindon-core

Core audio/ASR engine for the Larmindon real-time speech-to-text application.

This crate contains the platform-independent transcription pipeline: audio capture, resampling, voice activity detection (VAD), and streaming ASR via [parakeet-rs](https://github.com/altunenes/parakeet-rs) (Nemotron). It is designed to be consumed by different UI frontends through the `EngineEventSink` trait.

## Architecture

```
AudioEngine<E: EngineEventSink>
  ├── Audio capture (CPAL or PipeWire backend)
  ├── Resampling (48kHz → 16kHz via rubato)
  ├── VAD gating (Silero VAD v5 via ONNX Runtime)
  └── Streaming ASR (Nemotron via parakeet-rs)
```

### Key types

- **`EngineEventSink`** — Trait that frontends implement to receive transcription events. The Tauri app wraps `AppHandle::emit()`, a GTK app wraps `mpsc::Sender`, tests collect into a `Vec`.
- **`AudioEngine<E>`** — Main engine loop. Manages capture sessions, model caching, hot-reloadable settings, and diagnostics logging.
- **`VadProcessor`** / **`VadStateMachine`** — Voice activity detection with threshold hysteresis, leaky onset counting, and a pre-speech ring buffer.
- **`Settings`** — Configuration with file persistence, environment variable overrides, and validation.

## Usage

Add as a path dependency:

```toml
[dependencies]
larmindon-core = { path = "../larmindon-core" }
```

Implement `EngineEventSink` for your UI framework:

```rust
use larmindon_core::EngineEventSink;

#[derive(Clone)]
struct MyEventSink { /* ... */ }

impl EngineEventSink for MyEventSink {
    fn on_transcription(&self, text: String) { /* display text */ }
    fn on_error(&self, message: String) { /* show error */ }
    fn on_source_switched(&self, device_id: String) { /* update UI */ }
    fn on_devices_changed(&self, devices: Vec<AudioDevice>) { /* refresh list */ }
}
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `cpal` | Yes | Cross-platform audio capture via CPAL |
| `pipewire` | Yes | PipeWire audio capture (Linux only) |
| `webgpu` | No | WebGPU (Metal) execution provider for ASR |
| `directml` | No | DirectML execution provider for ASR (Windows) |
| `migraphx` | No | MIGraphX execution provider for ASR |

## Testing

```sh
cargo test          # Run all 58 tests
cargo test --lib    # Library tests only (no doc-tests)
```

Tests cover the VAD state machine, ring buffer, settings validation, punctuation detection, and device selection logic. Integration tests exercise the Silero VAD model with the bundled `models/silero_vad.onnx`.

## License

MIT
