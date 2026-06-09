# AGENTS.md

Cargo workspace for Larmindon's audio core and speech engines:

- `crates/larmindon-core` — capture backends, resampling, AGC, Silero VAD,
  the processing loop, settings, diagnostics, and the `SpeechEngine` trait +
  `EngineRegistry` abstraction.
- `crates/larmindon-engine-nemotron` — Nemotron (parakeet-rs) engine:
  fixed-chunk inference, punctuation/stuck-decoder resets. Always-final
  segments.
- `crates/larmindon-engine-april` — april-asr engine: streaming
  partial/final segments via a worker thread owning the `!Send` model.

## Architecture Notes

- Engines implement `SpeechEngine` (push-based `feed`/`on_speech_start`/
  `on_speech_end` + non-blocking `poll` for async result delivery) and emit
  `SegmentUpdate { segment_id, text, is_final }`. Core remaps engine-local
  segment ids to global ones (`engine/tracker.rs`).
- Engine crates provide an `EngineFactory` whose `EngineDescriptor` declares
  typed config fields (with optional env-var overrides); the app shell builds
  its Preferences UI from those descriptors. Core never depends on engine
  crates.
- AGC, VAD, and resampling are shared across engines and stay in core.

## Build Commands

- Prefix cargo commands that may download dependencies with `sfw` so they go
  through Socket Firewall Free. For example, use `sfw cargo fetch`,
  `sfw cargo install ...`, or `sfw cargo update` instead of running dependency
  fetch/install commands directly.
- Cargo commands that only use already-installed dependencies, such as
  `cargo fmt`, `cargo clippy`, `cargo check`, or `cargo test`, can be run
  directly unless they fail because dependencies are missing.
- `larmindon-engine-april` needs cmake + libclang at build time (it compiles
  vendored libaprilasr) and a system libonnxruntime at runtime
  (`brew install onnxruntime` on macOS).

## ONNX Runtime (important)

The april crate forces the `ort` crate into `load-dynamic` mode for any build
that includes it (one shared libonnxruntime per process; `ort`'s static
runtime cannot co-reside with libaprilasr's system one). Consequences:

- Workspace-wide builds/tests feature-unify `ort` into load-dynamic, so run
  tests with the dylib path set:

  ```sh
  ORT_DYLIB_PATH=$HOME/.config/larmindon/runtime/libonnxruntime.dylib cargo test --workspace
  ```

  (`~/.config/larmindon/runtime/libonnxruntime.dylib` is the locally
  re-signed copy maintained by the app's build script; macOS Gatekeeper
  blocks loading homebrew's foreign-ad-hoc-signed dylib directly.)
- Single-crate runs without the april crate (e.g.
  `cargo test -p larmindon-core`) use the static runtime and need nothing.

## Verification

Run from this directory:

```sh
cargo fmt
cargo clippy
cargo test --workspace   # with ORT_DYLIB_PATH, see above
```

End-to-end engine check without the app shell (feeds a 16 kHz float32 WAV
through the Nemotron engine):

```sh
say -o /tmp/test.wav --data-format=LEF32@16000 "Hello world."
cargo run -p larmindon-engine-nemotron --example feed_wav -- \
    ~/projects/nemotron-03-2026 /tmp/test.wav
```
