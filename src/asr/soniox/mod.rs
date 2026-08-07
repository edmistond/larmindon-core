//! Soniox cloud streaming ASR.
//!
//! Three layers, each testable on its own:
//!
//! * [`protocol`] — the wire types, error classification and PCM conversion.
//! * [`accumulator`] — the durable/volatile lanes that turn a revising token
//!   stream into `TranscriptUpdate`s.
//! * [`client`] — the socket thread, and [`backend`] the `AsrBackend` adapter
//!   that joins the two to the pipeline.

pub mod accumulator;
pub mod backend;
pub mod client;
pub mod protocol;

pub use accumulator::Accumulator;
pub use backend::SonioxBackend;
pub use client::{SonioxClient, SonioxConfig};
