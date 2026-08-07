//! Soniox cloud streaming ASR.
//!
//! Phase-scoped: the wire protocol and the accumulator are implemented and
//! tested offline here. The socket client that drives them is not wired up
//! yet, so `create_backend` does not offer this provider.

pub mod accumulator;
pub mod protocol;

pub use accumulator::Accumulator;
