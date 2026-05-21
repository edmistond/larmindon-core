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

## Verification

Run from this directory:

```sh
cargo fmt
cargo clippy
cargo test
```
