# Roadmap

## ASR Decoder Resilience

- Make the stuck-decoder reset threshold duration-based instead of chunk-count-based. The current `empty_reset_threshold` is counted in chunks, so changing `chunk_ms` also changes how much silent/empty decoder time is tolerated before reset. Prefer a millisecond setting, for example `empty_reset_threshold_ms`, and derive the chunk count with `ceil(threshold_ms / chunk_ms)` so 160ms, 560ms, and 1120ms modes use comparable reset timing.
