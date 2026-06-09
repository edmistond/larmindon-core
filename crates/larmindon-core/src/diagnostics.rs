use rusqlite::Connection;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Handle for diagnostics logging, shared between the core processing loop and
/// the active speech engine. Cloning is cheap; all clones write to the same
/// session row. A sink constructed with [`DiagSink::disabled`] (or when the
/// user starts a session with diagnostics off) makes every logging call a
/// no-op.
///
/// Writes are gated on a live toggle so flipping diagnostics off mid-session
/// stops writes while the connection stays open; re-enabling resumes writes to
/// the same session row.
#[derive(Clone)]
pub struct DiagSink {
    inner: Option<Arc<DiagInner>>,
}

struct DiagInner {
    conn: Mutex<Connection>,
    session_id: i64,
    enabled: Arc<AtomicBool>,
    epoch: Instant,
}

impl DiagSink {
    pub fn disabled() -> Self {
        Self { inner: None }
    }

    /// Open (or create) the diagnostics DB, run schema migrations, and insert
    /// a new session row. Call only when diagnostics are enabled at session
    /// start; pass the result to the engine via `SessionContext`.
    pub fn open(
        db_path: &Path,
        enabled: Arc<AtomicBool>,
        engine_id: &str,
        input_rate: usize,
        needs_resample: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        println!("[diag] Diagnostics DB: {}", db_path.display());
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS sessions (
                 id INTEGER PRIMARY KEY,
                 started_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now', 'localtime')),
                 input_rate INTEGER,
                 chunk_size INTEGER,
                 needs_resample INTEGER
             );
             CREATE TABLE IF NOT EXISTS events (
                 id INTEGER PRIMARY KEY,
                 session_id INTEGER,
                 ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now', 'localtime')),
                 uptime_ms INTEGER,
                 event_type TEXT,
                 chunk_num INTEGER,
                 inference_ms INTEGER,
                 drain_samples INTEGER,
                 drain_audio_ms REAL,
                 resample_in INTEGER,
                 resample_out INTEGER,
                 resample_leftover INTEGER,
                 asr_buf_len INTEGER,
                 text_empty INTEGER,
                 text_preview TEXT,
                 error_msg TEXT,
                 vad_state TEXT,
                 chunk_source TEXT
             );
             CREATE TABLE IF NOT EXISTS vad_events (
                 id INTEGER PRIMARY KEY,
                 session_id INTEGER,
                 ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now', 'localtime')),
                 uptime_ms INTEGER,
                 event_type TEXT,
                 pre_speech_samples INTEGER,
                 speech_duration_ms REAL,
                 consecutive_empty INTEGER,
                 probability REAL,
                 chunks_since_decoder_reset INTEGER,
                 audio_ms_since_decoder_reset INTEGER,
                 replay_chunks INTEGER,
                 replay_audio_ms INTEGER,
                 replay_nonempty_chunks INTEGER,
                 replay_inference_ms INTEGER
             );",
        )?;
        // Migrate: add columns if they don't exist (ALTER TABLE has no IF NOT EXISTS).
        migrate_add_column(&conn, "ALTER TABLE events ADD COLUMN vad_state TEXT;");
        migrate_add_column(&conn, "ALTER TABLE events ADD COLUMN vad_ms INTEGER;");
        migrate_add_column(&conn, "ALTER TABLE events ADD COLUMN resample_ms INTEGER;");
        migrate_add_column(&conn, "ALTER TABLE events ADD COLUMN iteration_ms INTEGER;");
        migrate_add_column(&conn, "ALTER TABLE events ADD COLUMN chunk_source TEXT;");
        migrate_add_column(
            &conn,
            "ALTER TABLE vad_events ADD COLUMN chunks_since_decoder_reset INTEGER;",
        );
        migrate_add_column(
            &conn,
            "ALTER TABLE vad_events ADD COLUMN audio_ms_since_decoder_reset INTEGER;",
        );
        migrate_add_column(
            &conn,
            "ALTER TABLE vad_events ADD COLUMN replay_chunks INTEGER;",
        );
        migrate_add_column(
            &conn,
            "ALTER TABLE vad_events ADD COLUMN replay_audio_ms INTEGER;",
        );
        migrate_add_column(
            &conn,
            "ALTER TABLE vad_events ADD COLUMN replay_nonempty_chunks INTEGER;",
        );
        migrate_add_column(
            &conn,
            "ALTER TABLE vad_events ADD COLUMN replay_inference_ms INTEGER;",
        );
        migrate_add_column(&conn, "ALTER TABLE sessions ADD COLUMN engine TEXT;");

        conn.execute(
            "INSERT INTO sessions (input_rate, needs_resample, engine) VALUES (?1, ?2, ?3)",
            rusqlite::params![input_rate as i64, needs_resample as i64, engine_id],
        )?;
        let session_id = conn.last_insert_rowid();

        Ok(Self {
            inner: Some(Arc::new(DiagInner {
                conn: Mutex::new(conn),
                session_id,
                enabled,
                epoch: Instant::now(),
            })),
        })
    }

    /// Milliseconds since the session started. Returns 0 when disabled.
    pub fn uptime_ms(&self) -> i64 {
        self.inner
            .as_ref()
            .map(|i| i.epoch.elapsed().as_millis() as i64)
            .unwrap_or(0)
    }

    /// Run `f` against the diagnostics connection with the current session id
    /// and uptime. No-op when the sink is disabled or the live toggle is off.
    /// The engine and the processing loop share one thread, so the mutex is
    /// uncontended.
    pub fn with_conn(&self, f: impl FnOnce(&Connection, i64, i64)) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        if !inner.enabled.load(Ordering::Relaxed) {
            return;
        }
        let uptime = inner.epoch.elapsed().as_millis() as i64;
        if let Ok(conn) = inner.conn.lock() {
            f(&conn, inner.session_id, uptime);
        }
    }

    /// Engines that consume fixed-size chunks record their chunk size on the
    /// session row; streaming engines leave it NULL.
    pub fn set_session_chunk_size(&self, chunk_size: usize) {
        self.with_conn(|conn, session_id, _| {
            let _ = conn.execute(
                "UPDATE sessions SET chunk_size = ?1 WHERE id = ?2",
                rusqlite::params![chunk_size as i64, session_id],
            );
        });
    }

    pub fn log_speech_start(&self, pre_speech_samples: usize) {
        self.with_conn(|conn, session_id, uptime| {
            let _ = conn.execute(
                "INSERT INTO vad_events (session_id, uptime_ms, event_type, pre_speech_samples)
                 VALUES (?1, ?2, 'speech_start', ?3)",
                rusqlite::params![session_id, uptime, pre_speech_samples as i64],
            );
        });
    }

    pub fn log_speech_end(&self, speech_duration_ms: f64) {
        self.with_conn(|conn, session_id, uptime| {
            let _ = conn.execute(
                "INSERT INTO vad_events (session_id, uptime_ms, event_type, speech_duration_ms)
                 VALUES (?1, ?2, 'speech_end', ?3)",
                rusqlite::params![session_id, uptime, speech_duration_ms],
            );
        });
    }

    /// One row per non-empty capture-buffer drain: how much audio came in and
    /// where the iteration time went. Engine-agnostic counterpart to the
    /// per-chunk 'transcribe' rows.
    pub fn log_feed(
        &self,
        drain_samples: usize,
        drain_audio_ms: f64,
        resample_ms: i64,
        vad_ms: i64,
        iteration_ms: i64,
    ) {
        self.with_conn(|conn, session_id, uptime| {
            let _ = conn.execute(
                "INSERT INTO events (session_id, uptime_ms, event_type, drain_samples,
                 drain_audio_ms, resample_ms, vad_ms, iteration_ms)
                 VALUES (?1, ?2, 'feed', ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    session_id,
                    uptime,
                    drain_samples as i64,
                    drain_audio_ms,
                    resample_ms,
                    vad_ms,
                    iteration_ms
                ],
            );
        });
    }

    pub fn log_error(&self, event_type: &str, error_msg: &str) {
        self.with_conn(|conn, session_id, uptime| {
            let _ = conn.execute(
                "INSERT INTO events (session_id, uptime_ms, event_type, error_msg)
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![session_id, uptime, event_type, error_msg],
            );
        });
    }

    pub fn log_shutdown(&self) {
        self.with_conn(|conn, session_id, uptime| {
            let _ = conn.execute(
                "INSERT INTO events (session_id, uptime_ms, event_type)
                 VALUES (?1, ?2, 'shutdown')",
                rusqlite::params![session_id, uptime],
            );
        });
    }
}

fn migrate_add_column(conn: &Connection, sql: &str) {
    if let Err(e) = conn.execute_batch(sql) {
        if !e.to_string().contains("duplicate column name") {
            eprintln!("[diag] Migration failed for `{}`: {}", sql, e);
        }
    }
}

/// Truncate text to at most `max_bytes` without splitting a UTF-8 character.
pub fn text_preview(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }

    let end = text
        .char_indices()
        .map(|(idx, _)| idx)
        .take_while(|&idx| idx <= max_bytes)
        .last()
        .unwrap_or(0);
    text[..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_preview_does_not_split_utf8() {
        assert_eq!(text_preview("abcédef", 4), "abc");
        assert_eq!(text_preview("abcédef", 5), "abcé");
    }
}
