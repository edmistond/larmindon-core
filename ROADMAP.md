# Roadmap

## ASR Decoder Resilience

- Make the stuck-decoder reset threshold duration-based instead of chunk-count-based. The current `empty_reset_threshold` is counted in chunks, so changing `chunk_ms` also changes how much silent/empty decoder time is tolerated before reset. Prefer a millisecond setting, for example `empty_reset_threshold_ms`, and derive the chunk count with `ceil(threshold_ms / chunk_ms)` so 160ms, 560ms, and 1120ms modes use comparable reset timing.

## Multi-Source ASR Model Sharing

- For future simultaneous capture, such as mic plus system/app audio or multiple sessions, use `parakeet-rs` shared model handles instead of loading one `Nemotron` per stream. Cache a `NemotronHandle` and spawn independent `Nemotron::from_shared` instances so each stream keeps separate decoder/audio state while sharing the expensive ONNX session. This should reduce memory and model-load cost, but inference through the shared model is still serialized by the model mutex, so it is an optimization for multi-source support rather than something needed for the current single active stream.
