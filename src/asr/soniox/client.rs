//! The Soniox realtime WebSocket client.
//!
//! One dedicated OS thread runs a `tokio` current-thread runtime; nothing async
//! escapes this module, so the rest of the app stays `std::thread` +
//! `std::sync::mpsc` + `Arc<AtomicBool>`. Two channels cross the boundary, and
//! their asymmetry is deliberate:
//!
//! * **Outbound** is a [`tokio::sync::mpsc::UnboundedSender`]. Its `send` is a
//!   plain synchronous method, so the processing thread can call it with no
//!   runtime and without blocking.
//! * **Inbound** is a [`std::sync::mpsc::Sender`]. It is `Send` and
//!   non-blocking, so the async side pushes freely and `process`/`poll` drain
//!   with `try_recv`.
//!
//! The socket is driven by a single `select!` over an **unsplit** stream rather
//! than `split()` into reader and writer tasks. One owner means one state
//! machine, which leaves the reconnect, finish and re-entrancy hazards below
//! nowhere to hide — a split reader could reconnect while the writer was still
//! sending into the dead socket.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc as tokio_mpsc;
use tokio_tungstenite::tungstenite::Message;

use super::protocol::{SonioxResponse, StartRequest, SONIOX_URL};
use crate::asr::AsrError;

/// WebSocket-level keepalive. The service has no application-level ping.
const PING_INTERVAL: Duration = Duration::from_secs(15);

/// How long teardown waits for the server's `finished` acknowledgement.
///
/// Nothing on the clean-shutdown path is bounded on its own, and a server that
/// goes quiet would hang teardown forever — a real, diagnosed field bug in the
/// reference client. Two seconds rather than five because `end_session` runs on
/// the processing thread, which `stop_active_session` then joins: a Start
/// issued right after a Stop queues behind this whole timeout. A dead Start
/// button is worse than a rarely-lost tail.
const FINISH_TIMEOUT: Duration = Duration::from_secs(2);

/// Bounds the TLS handshake, which is otherwise the one place a `Shutdown`
/// could not interrupt and so the one thing that could make `Drop`'s join
/// unbounded.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Consecutive failed attempts tolerated before the session is declared dead.
/// The counter resets whenever a connection lives long enough to return a
/// response, so this caps a burst of failures, not a long session's lifetime
/// total.
const MAX_RECONNECT_ATTEMPTS: u32 = 5;

const BACKOFF_START: Duration = Duration::from_millis(500);
const BACKOFF_CAP: Duration = Duration::from_secs(15);

/// Upload backlog ceiling, in samples at the pipeline's 16 kHz. The outbound
/// channel is unbounded, so a stalled socket would otherwise grow memory
/// without limit.
const MAX_QUEUED_SAMPLES: usize = 30 * 16_000;

/// Session parameters for one connection. Deliberately owns its own copies:
/// the socket thread outlives any borrow of `Settings`.
#[derive(Clone)]
pub struct SonioxConfig {
    pub api_key: String,
    pub model: String,
    pub language_hints: Vec<String>,
    pub diarization: bool,
    pub endpoint_detection: bool,
}

impl fmt::Debug for SonioxConfig {
    /// Hand-written for the same reason `Settings`' is: one `{:?}` away from
    /// putting the key in a log.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SonioxConfig")
            .field(
                "api_key",
                &if self.api_key.is_empty() {
                    "<unset>"
                } else {
                    "<redacted>"
                },
            )
            .field("model", &self.model)
            .field("language_hints", &self.language_hints)
            .field("diarization", &self.diarization)
            .field("endpoint_detection", &self.endpoint_detection)
            .finish()
    }
}

/// Processing thread → socket thread.
enum Outbound {
    Audio {
        pcm: Vec<u8>,
        samples: usize,
    },
    /// Begin the clean end-of-stream handshake.
    Finish,
    /// Drop everything now; used by `Drop`.
    Shutdown,
}

/// Socket thread → processing thread.
pub enum Inbound {
    /// A transcript frame to fold into the accumulator.
    Response(Box<SonioxResponse>),
    /// The connection dropped and is being re-established. The open segment
    /// must be finalized and the accumulator reset before the next response,
    /// because the new stream restarts its timeline from zero.
    Reconnecting(String),
    /// The server acknowledged end-of-stream, or teardown stopped waiting.
    Finished,
    /// The session cannot continue.
    Fatal(String),
}

/// Handle to the socket thread.
///
/// Dropping it sends a `Shutdown` and joins. Every await inside the socket task
/// is either bounded ([`CONNECT_TIMEOUT`], [`FINISH_TIMEOUT`]) or woken by that
/// `Shutdown`, so the join is bounded in practice without needing a timed join.
pub struct SonioxClient {
    outbound: tokio_mpsc::UnboundedSender<Outbound>,
    inbound: std_mpsc::Receiver<Inbound>,
    queued_samples: Arc<AtomicUsize>,
    thread: Option<JoinHandle<()>>,
    /// First layer of re-entrancy safety: a second `finish` sends nothing.
    finish_sent: bool,
}

impl SonioxClient {
    /// Spawns the socket thread and returns immediately. The connection itself
    /// is established asynchronously; failures surface as [`Inbound::Fatal`].
    pub fn connect(config: SonioxConfig) -> Result<Self, AsrError> {
        if config.api_key.trim().is_empty() {
            return Err(AsrError::Fatal(
                "No Soniox API key is configured. Add one in Preferences.".to_string(),
            ));
        }

        let (out_tx, out_rx) = tokio_mpsc::unbounded_channel();
        let (in_tx, in_rx) = std_mpsc::channel();
        let queued_samples = Arc::new(AtomicUsize::new(0));
        let queued_for_thread = Arc::clone(&queued_samples);

        let thread = thread::Builder::new()
            .name("soniox-socket".to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(e) => {
                        let _ = in_tx.send(Inbound::Fatal(format!(
                            "Could not start the Soniox runtime: {e}"
                        )));
                        return;
                    }
                };
                runtime.block_on(session(config, out_rx, in_tx, queued_for_thread));
            })
            .map_err(|e| {
                AsrError::Fatal(format!("Could not start the Soniox socket thread: {e}"))
            })?;

        Ok(Self {
            outbound: out_tx,
            inbound: in_rx,
            queued_samples,
            thread: Some(thread),
            finish_sent: false,
        })
    }

    /// Queues one PCM frame for upload.
    ///
    /// Rejects an empty buffer explicitly: an empty binary frame is the
    /// service's *end-of-stream* signal, so letting a zero-length chunk through
    /// would silently terminate the session mid-stream.
    pub fn send_audio(&self, pcm: Vec<u8>, samples: usize) -> Result<(), AsrError> {
        if pcm.is_empty() || samples == 0 {
            return Err(AsrError::Transient(
                "Refusing to send an empty Soniox audio frame; that is the end-of-stream signal"
                    .to_string(),
            ));
        }

        let queued = self.queued_samples.load(Ordering::Relaxed);
        if queued.saturating_add(samples) > MAX_QUEUED_SAMPLES {
            return Err(AsrError::Transient(format!(
                "Soniox upload is backed up past {}s of audio; dropping captured audio",
                MAX_QUEUED_SAMPLES / 16_000
            )));
        }

        self.queued_samples.fetch_add(samples, Ordering::Relaxed);
        self.outbound
            .send(Outbound::Audio { pcm, samples })
            .map_err(|_| {
                self.queued_samples.fetch_sub(samples, Ordering::Relaxed);
                AsrError::Fatal("The Soniox connection has ended".to_string())
            })
    }

    /// Everything that has arrived since the last call.
    pub fn drain(&self) -> Vec<Inbound> {
        let mut out = Vec::new();
        while let Ok(message) = self.inbound.try_recv() {
            out.push(message);
        }
        out
    }

    /// Begins the end-of-stream handshake. Idempotent.
    pub fn finish(&mut self) {
        if self.finish_sent {
            return;
        }
        self.finish_sent = true;
        let _ = self.outbound.send(Outbound::Finish);
    }

    /// Blocks for the next message, up to `timeout`. Used only by teardown,
    /// which must keep folding the finals the server flushes after EOS.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<Inbound> {
        self.inbound.recv_timeout(timeout).ok()
    }

    /// Samples accepted but not yet written to the socket.
    pub fn queued_samples(&self) -> usize {
        self.queued_samples.load(Ordering::Relaxed)
    }
}

impl Drop for SonioxClient {
    fn drop(&mut self) {
        let _ = self.outbound.send(Outbound::Shutdown);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Socket thread
// ---------------------------------------------------------------------------

/// Why one connection ended.
enum Outcome {
    /// End-of-stream completed, or teardown gave up waiting for it.
    Finished,
    /// A hard stop from `Drop`; say nothing more.
    Shutdown,
    Fatal(String),
    Retry {
        reason: String,
        /// Whether this connection got far enough to return a response. Used to
        /// reset the consecutive-failure budget.
        progressed: bool,
    },
}

/// Owns the reconnect loop. Audio captured while disconnected waits in
/// `pending` and goes out ahead of anything new once the socket is back.
async fn session(
    config: SonioxConfig,
    mut out_rx: tokio_mpsc::UnboundedReceiver<Outbound>,
    in_tx: std_mpsc::Sender<Inbound>,
    queued: Arc<AtomicUsize>,
) {
    let mut pending: VecDeque<(Vec<u8>, usize)> = VecDeque::new();
    let mut attempt: u32 = 0;

    loop {
        match connect_and_run(&config, &mut out_rx, &in_tx, &queued, &mut pending).await {
            Outcome::Finished => {
                let _ = in_tx.send(Inbound::Finished);
                return;
            }
            Outcome::Shutdown => return,
            Outcome::Fatal(message) => {
                let _ = in_tx.send(Inbound::Fatal(message));
                return;
            }
            Outcome::Retry { reason, progressed } => {
                if progressed {
                    attempt = 0;
                }
                attempt += 1;
                if attempt > MAX_RECONNECT_ATTEMPTS {
                    let _ = in_tx.send(Inbound::Fatal(format!(
                        "Lost the Soniox connection and could not re-establish it \
                         after {MAX_RECONNECT_ATTEMPTS} attempts: {reason}"
                    )));
                    return;
                }

                // The backend finalizes its open segment on this, because the
                // reconnected stream restarts its timeline from zero.
                let _ = in_tx.send(Inbound::Reconnecting(reason));

                match wait_backoff(backoff(attempt), &mut out_rx, &queued, &mut pending).await {
                    BackoffOutcome::Continue => {}
                    BackoffOutcome::Shutdown => return,
                    BackoffOutcome::Finish => {
                        // Stopped while disconnected: there is no socket to
                        // flush through, so teardown is already complete.
                        let _ = in_tx.send(Inbound::Finished);
                        return;
                    }
                }
            }
        }
    }
}

/// 0.5 → 1 → 2 → 4 → 8 s, capped.
fn backoff(attempt: u32) -> Duration {
    BACKOFF_START
        .saturating_mul(1u32 << attempt.saturating_sub(1).min(16))
        .min(BACKOFF_CAP)
}

enum BackoffOutcome {
    Continue,
    Shutdown,
    Finish,
}

/// Sleeps out the backoff while still accepting audio, so a reconnect does not
/// punch a hole in the stream, and while staying interruptible by `Shutdown`.
async fn wait_backoff(
    delay: Duration,
    out_rx: &mut tokio_mpsc::UnboundedReceiver<Outbound>,
    queued: &Arc<AtomicUsize>,
    pending: &mut VecDeque<(Vec<u8>, usize)>,
) -> BackoffOutcome {
    let deadline = tokio::time::Instant::now() + delay;
    loop {
        tokio::select! {
            _ = tokio::time::sleep_until(deadline) => return BackoffOutcome::Continue,
            message = out_rx.recv() => match message {
                None | Some(Outbound::Shutdown) => return BackoffOutcome::Shutdown,
                Some(Outbound::Finish) => return BackoffOutcome::Finish,
                Some(Outbound::Audio { pcm, samples }) => {
                    // Honour the same ceiling the sender applies: it counts a
                    // frame as queued the moment it is accepted, and these are
                    // still queued.
                    if queued.load(Ordering::Relaxed) > MAX_QUEUED_SAMPLES {
                        queued.fetch_sub(samples, Ordering::Relaxed);
                    } else {
                        pending.push_back((pcm, samples));
                    }
                }
            },
        }
    }
}

/// Writes one frame and releases its share of the backlog. The counter is
/// decremented whether or not the write succeeds, so a dying socket cannot leak
/// the budget.
async fn send_frame<S>(
    ws: &mut tokio_tungstenite::WebSocketStream<S>,
    pcm: Vec<u8>,
    samples: usize,
    queued: &Arc<AtomicUsize>,
) -> Result<(), tokio_tungstenite::tungstenite::Error>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    queued.fetch_sub(
        samples.min(queued.load(Ordering::Relaxed)),
        Ordering::Relaxed,
    );
    ws.send(Message::binary(pcm)).await
}

/// One connection, start to finish.
async fn connect_and_run(
    config: &SonioxConfig,
    out_rx: &mut tokio_mpsc::UnboundedReceiver<Outbound>,
    in_tx: &std_mpsc::Sender<Inbound>,
    queued: &Arc<AtomicUsize>,
    pending: &mut VecDeque<(Vec<u8>, usize)>,
) -> Outcome {
    let mut ws = match tokio::time::timeout(
        CONNECT_TIMEOUT,
        tokio_tungstenite::connect_async(SONIOX_URL),
    )
    .await
    {
        Err(_) => {
            return Outcome::Retry {
                reason: "the connection to Soniox timed out".to_string(),
                progressed: false,
            }
        }
        Ok(Err(e)) => {
            return Outcome::Retry {
                reason: format!("could not reach Soniox: {e}"),
                progressed: false,
            }
        }
        Ok(Ok((ws, _response))) => ws,
    };

    // Auth is the api_key field of this first frame; there is no header auth.
    let start = StartRequest::new(
        config.api_key.clone(),
        config.model.clone(),
        config.language_hints.clone(),
        config.diarization,
        config.endpoint_detection,
    );
    let start_json = match serde_json::to_string(&start) {
        Ok(json) => json,
        Err(e) => return Outcome::Fatal(format!("Could not build the Soniox start request: {e}")),
    };
    if let Err(e) = ws.send(Message::text(start_json)).await {
        return Outcome::Retry {
            reason: format!("could not start a Soniox session: {e}"),
            progressed: false,
        };
    }

    // Audio buffered across the outage goes first, in capture order.
    while let Some((pcm, samples)) = pending.pop_front() {
        if let Err(e) = send_frame(&mut ws, pcm, samples, queued).await {
            return Outcome::Retry {
                reason: format!("Soniox audio upload failed: {e}"),
                progressed: false,
            };
        }
    }

    let mut progressed = false;
    let mut finishing = false;

    let mut ping = tokio::time::interval(PING_INTERVAL);
    // The first tick resolves immediately; consume it so the keepalive does not
    // fire the instant the socket opens.
    ping.tick().await;

    // Armed only once end-of-stream has been sent.
    let watchdog = tokio::time::sleep(Duration::ZERO);
    tokio::pin!(watchdog);
    let mut watchdog_armed = false;

    loop {
        tokio::select! {
            outgoing = out_rx.recv() => match outgoing {
                // The handle was dropped without a Shutdown reaching us.
                None => return Outcome::Shutdown,

                Some(Outbound::Shutdown) => {
                    let _ = ws.close(None).await;
                    return Outcome::Shutdown;
                }

                // Second layer of re-entrancy safety: a second stop arriving
                // mid-finish must not cancel the socket, or the finals the
                // server is still flushing are discarded.
                Some(Outbound::Finish) if finishing => {}

                Some(Outbound::Finish) => {
                    // Everything already queued must go out BEFORE the
                    // end-of-stream frame, or the tail is truncated. The sender
                    // queued all of it ahead of this message, so it is all
                    // sitting in the channel right now.
                    loop {
                        match out_rx.try_recv() {
                            Ok(Outbound::Audio { pcm, samples }) => {
                                if let Err(e) = send_frame(&mut ws, pcm, samples, queued).await {
                                    return Outcome::Retry {
                                        reason: format!("Soniox audio upload failed: {e}"),
                                        progressed,
                                    };
                                }
                            }
                            Ok(Outbound::Finish) => {}
                            Ok(Outbound::Shutdown) => {
                                let _ = ws.close(None).await;
                                return Outcome::Shutdown;
                            }
                            Err(_) => break,
                        }
                    }

                    // An empty binary frame IS the end-of-stream signal.
                    if ws.send(Message::binary(Vec::new())).await.is_err() {
                        // The socket is already gone; there is nothing left to
                        // flush and nothing to wait for.
                        return Outcome::Finished;
                    }
                    finishing = true;
                    watchdog_armed = true;
                    watchdog
                        .as_mut()
                        .reset(tokio::time::Instant::now() + FINISH_TIMEOUT);
                }

                Some(Outbound::Audio { pcm, samples }) => {
                    if finishing {
                        // End-of-stream is already on the wire; anything after
                        // it is not part of this stream.
                        let _ = pcm;
                        queued.fetch_sub(
                            samples.min(queued.load(Ordering::Relaxed)),
                            Ordering::Relaxed,
                        );
                    } else if let Err(e) = send_frame(&mut ws, pcm, samples, queued).await {
                        return Outcome::Retry {
                            reason: format!("Soniox audio upload failed: {e}"),
                            progressed,
                        };
                    }
                }
            },

            incoming = ws.next() => match incoming {
                None => {
                    return if finishing {
                        Outcome::Finished
                    } else {
                        Outcome::Retry {
                            reason: "Soniox closed the connection".to_string(),
                            progressed,
                        }
                    };
                }

                Some(Err(e)) => {
                    return if finishing {
                        Outcome::Finished
                    } else {
                        Outcome::Retry {
                            reason: format!("the Soniox connection dropped: {e}"),
                            progressed,
                        }
                    };
                }

                Some(Ok(Message::Text(text))) => {
                    let response: SonioxResponse = match serde_json::from_str(text.as_str()) {
                        Ok(response) => response,
                        Err(e) => {
                            // Every field is optional and the shape is
                            // additive, so this means a genuinely malformed
                            // frame rather than a field we do not know yet.
                            // Skipping beats killing a working session.
                            eprintln!("[soniox] ignoring unparseable frame: {e}");
                            continue;
                        }
                    };
                    progressed = true;

                    // Errors are told apart by error_type being present — never
                    // by error_code, and never by an empty token list, which is
                    // normal during silence.
                    if let Some(error) = response.error() {
                        let reason = format!("{} ({})", error.message, error.error_type);
                        return if error.retryable {
                            Outcome::Retry { reason, progressed }
                        } else {
                            Outcome::Fatal(format!("Soniox rejected the session: {reason}"))
                        };
                    }

                    let finished = response.finished;
                    if !response.tokens.is_empty() {
                        let _ = in_tx.send(Inbound::Response(Box::new(response)));
                    }
                    if finished {
                        let _ = ws.close(None).await;
                        return Outcome::Finished;
                    }
                }

                // Responses are text frames only; a binary frame from the
                // server is a protocol violation, not a hiccup.
                Some(Ok(Message::Binary(_))) => {
                    return Outcome::Fatal(
                        "Soniox sent a binary frame, which its protocol never uses".to_string(),
                    );
                }

                Some(Ok(Message::Close(_))) => {
                    return if finishing {
                        Outcome::Finished
                    } else {
                        Outcome::Retry {
                            reason: "Soniox closed the connection".to_string(),
                            progressed,
                        }
                    };
                }

                // Ping/Pong/Frame: tungstenite answers pings itself.
                Some(Ok(_)) => {}
            },

            _ = ping.tick() => {
                if let Err(e) = ws.send(Message::Ping(Vec::new().into())).await {
                    return Outcome::Retry {
                        reason: format!("the Soniox keepalive failed: {e}"),
                        progressed,
                    };
                }
            }

            _ = &mut watchdog, if watchdog_armed => {
                // The server went quiet after end-of-stream. Force the close
                // and report success: the cancellation error is suppressed
                // because a rarely-lost tail beats a Start button that stays
                // dead while stop_active_session joins the processing thread.
                let _ = ws.close(None).await;
                return Outcome::Finished;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_and_caps() {
        assert_eq!(backoff(1), Duration::from_millis(500));
        assert_eq!(backoff(2), Duration::from_secs(1));
        assert_eq!(backoff(3), Duration::from_secs(2));
        assert_eq!(backoff(4), Duration::from_secs(4));
        assert_eq!(backoff(5), Duration::from_secs(8));
        // Never past the cap, however many attempts are asked for.
        assert_eq!(backoff(6), BACKOFF_CAP);
        assert_eq!(backoff(60), BACKOFF_CAP);
    }

    #[test]
    fn config_debug_never_prints_the_key() {
        let config = SonioxConfig {
            api_key: "super-secret".to_string(),
            model: "stt-rt-v5".to_string(),
            language_hints: vec!["en".to_string()],
            diarization: true,
            endpoint_detection: true,
        };
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("super-secret"));
        assert!(rendered.contains("<redacted>"));

        let unset = SonioxConfig {
            api_key: String::new(),
            ..config
        };
        assert!(format!("{unset:?}").contains("<unset>"));
    }

    #[test]
    fn connect_rejects_a_blank_key_without_spawning_a_thread() {
        let config = SonioxConfig {
            api_key: "   ".to_string(),
            model: "stt-rt-v5".to_string(),
            language_hints: vec!["en".to_string()],
            diarization: true,
            endpoint_detection: true,
        };
        match SonioxClient::connect(config) {
            Err(AsrError::Fatal(message)) => assert!(message.contains("API key")),
            _ => panic!("a blank key must be rejected as fatal"),
        }
    }
}
