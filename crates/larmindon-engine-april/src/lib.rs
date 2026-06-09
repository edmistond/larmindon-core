//! april-asr speech engine.
//!
//! April is a streaming model with revisable hypotheses: it emits
//! `RecognitionPartial` updates that each replace the current utterance text,
//! then a `RecognitionFinal` that closes it. This maps directly onto the
//! transient/final [`SegmentUpdate`] model — one open segment at a time whose
//! text is rewritten until finalized.
//!
//! Two constraints of the `aprilasr` bindings shape this implementation:
//!
//! - `Session::new` takes a plain `fn` pointer, so results are routed through
//!   a process-global channel slot ([`RESULT_TX`]). Only one April session
//!   exists per process (the app runs one engine at a time).
//! - `Model` and `Session<'_>` hold raw pointers (`!Send`) and the session
//!   borrows the model, so both live on a dedicated worker thread that owns
//!   them for its whole stack frame; the engine talks to it over channels.
//!   This also keeps the model loaded across sessions for instant restarts.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Mutex, Once};
use std::thread::JoinHandle;

use aprilasr::{init_april_api, Model, ResultType, Session};
use serde::{Deserialize, Serialize};

use larmindon_core::diagnostics::{text_preview, DiagSink};
use larmindon_core::engine::registry::{
    ConfigField, EngineDescriptor, EngineFactory, EngineKind, FieldType,
};
use larmindon_core::engine::{EngineError, SegmentUpdate, SessionContext, SpeechEngine};
use larmindon_core::settings::expand_tilde;

pub const ENGINE_ID: &str = "april";
const APRIL_API_VERSION: i32 = 1;
const ASR_SAMPLE_RATE: usize = 16000;

/// With `ort/load-dynamic` active (which this crate forces — see Cargo.toml),
/// ONNX Runtime is dlopen'd at first use from `ORT_DYLIB_PATH`. Call this
/// once at app startup, before any model loads. No-op if the user already
/// set the variable.
///
/// On macOS the preferred target is the locally re-signed copy that the app's
/// build script maintains under `~/.config/larmindon/runtime/` — Gatekeeper
/// blocks the package manager's foreign-ad-hoc-signed dylib ("Apple could not
/// verify..."), and our patched libaprilasr loads the same managed copy, so
/// pointing ort at it keeps exactly one ONNX Runtime in the process.
pub fn ensure_ort_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }

    let managed = larmindon_core::settings::Settings::config_dir()
        .join("runtime")
        .join("libonnxruntime.dylib");
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if cfg!(target_os = "macos") {
        candidates.push(managed);
    }
    candidates.extend(
        [
            "/opt/homebrew/lib/libonnxruntime.dylib",
            "/usr/local/lib/libonnxruntime.dylib",
            "/usr/lib/libonnxruntime.so",
            "/usr/local/lib/libonnxruntime.so",
        ]
        .iter()
        .map(std::path::PathBuf::from),
    );

    let Some(resolved) = candidates.into_iter().find(|path| path.exists()) else {
        eprintln!(
            "Warning: no libonnxruntime found and ORT_DYLIB_PATH unset; \
             ONNX-based engines may fail to initialize"
        );
        return;
    };

    println!("Using ONNX Runtime dylib at {}", resolved.display());
    std::env::set_var("ORT_DYLIB_PATH", &resolved);
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AprilConfig {
    /// Path to the `.april` model file.
    pub model_path: String,
}

impl AprilConfig {
    fn parse(config: &serde_json::Value) -> Result<Self, String> {
        let cfg: Self = serde_json::from_value(config.clone())
            .map_err(|e| format!("Invalid april config: {}", e))?;
        if cfg.model_path.trim().is_empty() {
            return Err("model_path cannot be empty".to_string());
        }
        Ok(cfg)
    }
}

pub struct AprilFactory;

impl EngineFactory for AprilFactory {
    fn descriptor(&self) -> EngineDescriptor {
        EngineDescriptor {
            id: ENGINE_ID,
            name: "April ASR",
            kind: EngineKind::Local,
            emits_partials: true,
            config_fields: vec![ConfigField {
                key: "model_path",
                label: "Model File",
                field: FieldType::Path { directory: false },
                default: "".into(),
                env_var: None,
                help: Some("Path to a .april model file (e.g. aprilv0_en-us.april)"),
            }],
        }
    }

    fn default_config(&self) -> serde_json::Value {
        serde_json::to_value(AprilConfig::default()).expect("config serializes")
    }

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), String> {
        AprilConfig::parse(config).map(|_| ())
    }

    fn cache_key(&self, config: &serde_json::Value) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let cfg = AprilConfig::parse(config).unwrap_or_default();
        let mut hasher = DefaultHasher::new();
        expand_tilde(&cfg.model_path)
            .to_string_lossy()
            .hash(&mut hasher);
        hasher.finish()
    }

    fn create(&self, config: &serde_json::Value) -> Result<Box<dyn SpeechEngine>, EngineError> {
        let cfg = AprilConfig::parse(config).map_err(EngineError::Fatal)?;
        Ok(Box::new(AprilEngine::new(cfg)))
    }
}

/// (is_final, joined token text)
type AprilResult = (bool, String);

/// The session callback is a plain `fn` pointer, so this global slot routes
/// results to the active engine instance. One April session per process.
static RESULT_TX: Mutex<Option<Sender<AprilResult>>> = Mutex::new(None);

fn april_callback(result: ResultType) {
    let mapped = match result {
        ResultType::RecognitionPartial(tokens) => Some((false, join_tokens(&tokens))),
        ResultType::RecognitionFinal(tokens) => Some((true, join_tokens(&tokens))),
        _ => None,
    };
    if let Some(update) = mapped {
        if let Ok(guard) = RESULT_TX.lock() {
            if let Some(tx) = guard.as_ref() {
                let _ = tx.send(update);
            }
        }
    }
}

fn join_tokens(tokens: &[aprilasr::Token]) -> String {
    tokens.iter().map(|t| t.token()).collect()
}

struct OpenRequest {
    model_path: String,
    reply: Sender<Result<(), String>>,
}

enum WorkerCmd {
    Open(OpenRequest),
    Feed(Vec<u8>),
    Flush,
    /// Drop the session but keep the model loaded for the next session.
    CloseSession,
    Shutdown,
}

/// Owns the `!Send` Model + Session. The model lives in the outer loop frame;
/// the session borrows it inside, which is exactly the lifetime relationship
/// the binding requires. A changed model path breaks out to reload.
fn worker_main(cmd_rx: Receiver<WorkerCmd>) {
    static INIT: Once = Once::new();
    INIT.call_once(|| init_april_api(APRIL_API_VERSION));

    let mut pending_open: Option<OpenRequest> = None;
    loop {
        let open = match pending_open.take() {
            Some(open) => open,
            None => match cmd_rx.recv() {
                Ok(WorkerCmd::Open(open)) => open,
                Ok(WorkerCmd::Shutdown) | Err(_) => return,
                Ok(_) => continue, // Feed/Flush/Close with no session: ignore
            },
        };

        let model = match Model::new(&open.model_path) {
            Ok(model) => model,
            Err(e) => {
                let _ = open
                    .reply
                    .send(Err(format!("Failed to load april model: {}", e)));
                continue;
            }
        };
        if model.sample_rate() != ASR_SAMPLE_RATE {
            let _ = open.reply.send(Err(format!(
                "April model '{}' wants {} Hz; the pipeline is fixed at {} Hz",
                model.name(),
                model.sample_rate(),
                ASR_SAMPLE_RATE
            )));
            continue;
        }
        println!(
            "April model '{}' loaded ({}, {} Hz)",
            model.name(),
            model.language(),
            model.sample_rate()
        );
        let model_path = open.model_path.clone();

        let mut session = match Session::new(&model, april_callback, true, true) {
            Ok(session) => {
                let _ = open.reply.send(Ok(()));
                Some(session)
            }
            Err(e) => {
                let _ = open
                    .reply
                    .send(Err(format!("Failed to create april session: {}", e)));
                continue;
            }
        };

        loop {
            match cmd_rx.recv() {
                Ok(WorkerCmd::Feed(bytes)) => {
                    if let Some(session) = session.as_ref() {
                        session.feed_pcm16(&bytes);
                    }
                }
                Ok(WorkerCmd::Flush) => {
                    if let Some(session) = session.as_ref() {
                        session.flush();
                    }
                }
                Ok(WorkerCmd::CloseSession) => {
                    session = None;
                }
                Ok(WorkerCmd::Open(open)) => {
                    if open.model_path == model_path {
                        // Same model: just start a fresh session on it.
                        drop(session.take());
                        match Session::new(&model, april_callback, true, true) {
                            Ok(new_session) => {
                                let _ = open.reply.send(Ok(()));
                                session = Some(new_session);
                            }
                            Err(e) => {
                                let _ = open
                                    .reply
                                    .send(Err(format!("Failed to create april session: {}", e)));
                            }
                        }
                    } else {
                        drop(session.take());
                        pending_open = Some(open);
                        break; // drop this model, reload in the outer loop
                    }
                }
                Ok(WorkerCmd::Shutdown) | Err(_) => return,
            }
        }
    }
}

pub struct AprilEngine {
    config: AprilConfig,
    worker_tx: Option<Sender<WorkerCmd>>,
    worker: Option<JoinHandle<()>>,
    results_rx: Option<Receiver<AprilResult>>,
    /// Engine-local id of the in-flight (partial) segment, if any.
    current_segment: Option<u64>,
    /// Last partial text seen for the open segment, used to force-finalize on
    /// teardown if the closing final never arrives.
    last_partial_text: String,
    next_segment_id: u64,
    final_count: u64,
    diag: DiagSink,
}

impl AprilEngine {
    fn new(config: AprilConfig) -> Self {
        Self {
            config,
            worker_tx: None,
            worker: None,
            results_rx: None,
            current_segment: None,
            last_partial_text: String::new(),
            next_segment_id: 0,
            final_count: 0,
            diag: DiagSink::disabled(),
        }
    }

    fn send(&self, cmd: WorkerCmd) -> Result<(), EngineError> {
        self.worker_tx
            .as_ref()
            .ok_or_else(|| EngineError::Fatal("april worker not running".to_string()))?
            .send(cmd)
            .map_err(|_| EngineError::Fatal("april worker thread died".to_string()))
    }

    /// Drain queued results into segment updates. Multiple partials drained in
    /// one tick are coalesced down to the latest (each replaces the previous
    /// hypothesis wholesale); finals are kept in order.
    fn drain_results(&mut self) -> Vec<SegmentUpdate> {
        let Some(rx) = self.results_rx.as_ref() else {
            return Vec::new();
        };
        let mut received: Vec<AprilResult> = Vec::new();
        while let Ok(result) = rx.try_recv() {
            received.push(result);
        }
        self.apply_results(received)
    }

    fn apply_results(&mut self, received: Vec<AprilResult>) -> Vec<SegmentUpdate> {
        let mut updates = Vec::new();
        let mut pending_partial: Option<String> = None;

        for (is_final, text) in received {
            if is_final {
                // The final supersedes any earlier partial drained this tick.
                pending_partial = None;
                self.last_partial_text.clear();
                let segment_id = self.current_segment.take().unwrap_or_else(|| {
                    let id = self.next_segment_id;
                    self.next_segment_id += 1;
                    id
                });
                self.final_count += 1;
                self.log_result(&text, true);
                if !text.is_empty() {
                    updates.push(SegmentUpdate {
                        segment_id,
                        text,
                        is_final: true,
                    });
                }
            } else {
                pending_partial = Some(text);
            }
        }

        if let Some(text) = pending_partial {
            let segment_id = *self.current_segment.get_or_insert_with(|| {
                let id = self.next_segment_id;
                self.next_segment_id += 1;
                id
            });
            self.last_partial_text = text.clone();
            updates.push(SegmentUpdate {
                segment_id,
                text,
                is_final: false,
            });
        }

        updates
    }

    fn log_result(&self, text: &str, is_final: bool) {
        let preview = text_preview(text, 200);
        let chunk_num = self.final_count;
        let source = if is_final { "final" } else { "partial" };
        let is_empty = text.is_empty();
        self.diag.with_conn(|conn, session_id, uptime| {
            let _ = conn.execute(
                "INSERT INTO events (session_id, uptime_ms, event_type, chunk_num,
                 text_empty, text_preview, chunk_source)
                 VALUES (?1, ?2, 'transcribe', ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    session_id,
                    uptime,
                    chunk_num as i64,
                    is_empty as i64,
                    preview,
                    source,
                ],
            );
        });
    }

    /// Force-finalize a dangling partial segment (used at teardown when the
    /// closing final never arrived).
    fn force_finalize(&mut self) -> Option<SegmentUpdate> {
        let segment_id = self.current_segment.take()?;
        let text = std::mem::take(&mut self.last_partial_text);
        if text.is_empty() {
            return None;
        }
        Some(SegmentUpdate {
            segment_id,
            text,
            is_final: true,
        })
    }
}

impl SpeechEngine for AprilEngine {
    fn engine_id(&self) -> &'static str {
        ENGINE_ID
    }

    fn begin_session(
        &mut self,
        ctx: SessionContext,
        config: &serde_json::Value,
    ) -> Result<(), EngineError> {
        let cfg = AprilConfig::parse(config).map_err(EngineError::Fatal)?;
        self.config = cfg;
        self.diag = ctx.diag;
        self.current_segment = None;
        self.last_partial_text.clear();

        if self.worker_tx.is_none() {
            let (tx, rx) = channel();
            self.worker = Some(
                std::thread::Builder::new()
                    .name("april-worker".to_string())
                    .spawn(move || worker_main(rx))
                    .map_err(|e| EngineError::Fatal(format!("spawn april worker: {}", e)))?,
            );
            self.worker_tx = Some(tx);
        }

        // Fresh result channel per session so stale results can't leak in.
        let (result_tx, result_rx) = channel();
        *RESULT_TX.lock().unwrap() = Some(result_tx);
        self.results_rx = Some(result_rx);

        let model_path = expand_tilde(&self.config.model_path)
            .to_string_lossy()
            .to_string();
        let (reply_tx, reply_rx) = channel();
        self.send(WorkerCmd::Open(OpenRequest {
            model_path,
            reply: reply_tx,
        }))?;
        reply_rx
            .recv()
            .map_err(|_| EngineError::Fatal("april worker thread died".to_string()))?
            .map_err(EngineError::Fatal)
    }

    fn on_speech_start(&mut self) {}

    fn feed(&mut self, samples: &[f32]) -> Result<Vec<SegmentUpdate>, EngineError> {
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &sample in samples {
            let pcm = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            bytes.extend_from_slice(&pcm.to_le_bytes());
        }
        self.send(WorkerCmd::Feed(bytes))?;
        Ok(self.drain_results())
    }

    fn on_speech_end(&mut self) -> Result<Vec<SegmentUpdate>, EngineError> {
        // Flush forces the in-flight hypothesis to finalize; the final arrives
        // asynchronously via poll().
        self.send(WorkerCmd::Flush)?;
        Ok(self.drain_results())
    }

    fn poll(&mut self) -> Result<Vec<SegmentUpdate>, EngineError> {
        Ok(self.drain_results())
    }

    fn update_config(&mut self, config: &serde_json::Value) {
        // model_path is covered by cache_key (engine recreated on change);
        // nothing hot-reloadable yet.
        if let Ok(cfg) = AprilConfig::parse(config) {
            self.config = cfg;
        }
    }

    fn end_session(&mut self) -> Result<Vec<SegmentUpdate>, EngineError> {
        let _ = self.send(WorkerCmd::Flush);
        // Give the flush a moment to deliver the closing final.
        let mut updates = Vec::new();
        if let Some(rx) = self.results_rx.as_ref() {
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(300);
            let mut received = Vec::new();
            while std::time::Instant::now() < deadline {
                match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                    Ok(result) => {
                        let done = result.0;
                        received.push(result);
                        if done {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            updates = self.apply_results(received);
        }
        updates.extend(self.force_finalize());

        let _ = self.send(WorkerCmd::CloseSession);
        *RESULT_TX.lock().unwrap() = None;
        self.results_rx = None;
        self.diag = DiagSink::disabled();
        Ok(updates)
    }
}

impl Drop for AprilEngine {
    fn drop(&mut self) {
        if let Some(tx) = self.worker_tx.take() {
            let _ = tx.send(WorkerCmd::Shutdown);
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> AprilEngine {
        AprilEngine::new(AprilConfig {
            model_path: "/m.april".to_string(),
        })
    }

    #[test]
    fn partials_update_one_segment_until_final() {
        let mut e = engine();

        let updates = e.apply_results(vec![(false, "HE".into()), (false, "HELLO".into())]);
        assert_eq!(updates.len(), 1, "partials coalesce to the latest");
        assert_eq!(updates[0].text, "HELLO");
        assert!(!updates[0].is_final);
        let open_id = updates[0].segment_id;

        let updates = e.apply_results(vec![(false, "HELLO WO".into())]);
        assert_eq!(updates[0].segment_id, open_id, "same segment while open");

        let updates = e.apply_results(vec![(true, "HELLO WORLD".into())]);
        assert_eq!(updates[0].segment_id, open_id);
        assert!(updates[0].is_final);

        let updates = e.apply_results(vec![(false, "NEXT".into())]);
        assert_ne!(updates[0].segment_id, open_id, "final closed the segment");
    }

    #[test]
    fn final_supersedes_partials_in_same_tick() {
        let mut e = engine();
        let updates = e.apply_results(vec![
            (false, "HELLO WOR".into()),
            (true, "HELLO WORLD".into()),
            (false, "NE".into()),
        ]);
        assert_eq!(updates.len(), 2);
        assert!(updates[0].is_final);
        assert_eq!(updates[0].text, "HELLO WORLD");
        assert!(!updates[1].is_final);
        assert_eq!(updates[1].text, "NE");
        assert_ne!(updates[0].segment_id, updates[1].segment_id);
    }

    #[test]
    fn empty_final_closes_segment_without_update() {
        let mut e = engine();
        e.apply_results(vec![(false, "X".into())]);
        let updates = e.apply_results(vec![(true, "".into())]);
        assert!(updates.is_empty());
        assert!(e.current_segment.is_none());
    }

    #[test]
    fn force_finalize_emits_last_partial() {
        let mut e = engine();
        e.apply_results(vec![(false, "DANGLING".into())]);
        let update = e.force_finalize().expect("dangling partial finalized");
        assert!(update.is_final);
        assert_eq!(update.text, "DANGLING");
        assert!(e.force_finalize().is_none());
    }
}
