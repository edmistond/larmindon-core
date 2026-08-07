# AGENTS.md

Rust core library for Larmindon audio capture, VAD, ASR orchestration, and
settings logic.

## Build Commands

- Prefix cargo commands that may download dependencies with `sfw` so they go
  through Socket Firewall Free. For example, use `sfw cargo fetch`,
  `sfw cargo install ...`, or `sfw cargo update` instead of running dependency
  fetch/install commands directly.
- Cargo commands that only use already-installed dependencies, such as
  `cargo fmt`, `cargo clippy`, `cargo check`, or `cargo test`, can be run
  directly unless they fail because dependencies are missing.

- `sfw cargo fetch` currently fails on this machine with a schannel
  `CERT_TRUST_REVOCATION_STATUS_UNKNOWN` error, because Windows cannot do a
  revocation lookup for the proxy's generated CA. Prefix with
  `CARGO_HTTP_CHECK_REVOKE=false` rather than dropping `sfw`; certificate
  validation still happens, only the revocation check is skipped.

## Verification

Run from this directory:

```sh
cargo fmt
cargo clippy
cargo test
```

After changing anything in `src/asr/`, `src/audio_engine.rs`, `src/vad.rs` or
`src/agc.rs`, also run the ASR behaviour-identity gate:

```sh
bash testdata/check_regression.sh
```

It replays a fixed WAV through the real engine and model and diffs the
transcript and diagnostics against committed goldens. `processing_loop` has no
unit-test seam, so this is the only check that catches a transcript regression.
See `testdata/README.md` for what legitimately changes the goldens.
