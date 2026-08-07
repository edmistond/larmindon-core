//! SQLite diagnostics for the processing loop.
//!
//! Every write is gated on a live `Arc<AtomicBool>`: flipping diagnostics off
//! mid-session skips writes while the connection and session row stay alive, so
//! re-enabling resumes writing to the same session.
//!
//! `uptime_ms` is always supplied by the caller rather than computed here.
//! Several call sites capture a timestamp before mutating state and write the
//! row afterwards (the mid-speech reset captures it before `model.reset()` and
//! the replay loop), so taking the clock reading here would be subtly wrong.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rusqlite::Connection;

/// One `transcribe` row. `chunk_source` distinguishes live chunks from ones
/// re-run through the decoder after a mid-speech reset.
pub struct TranscribeRow<'a> {
    pub uptime_ms: i64,
    pub chunk_num: u64,
    pub inference_ms: i64,
    pub drain_samples: usize,
    pub drain_audio_ms: f64,
    pub asr_buf_len: usize,
    pub is_empty: bool,
    pub preview: &'a str,
    pub vad_state: &'a str,
    pub vad_ms: i64,
    pub resample_ms: i64,
    pub iteration_ms: i64,
    pub chunk_source: &'a str,
}

pub struct AsrErrorRow<'a> {
    pub uptime_ms: i64,
    pub chunk_num: u64,
    pub inference_ms: i64,
    pub error_msg: &'a str,
    pub vad_state: &'a str,
    pub chunk_source: &'a str,
}

pub struct PunctuationResetRow {
    pub uptime_ms: i64,
    pub consecutive_empty: u32,
    pub chunks_since_decoder_reset: u64,
}

pub struct MidSpeechResetRow {
    pub uptime_ms: i64,
    pub consecutive_empty: u32,
    pub chunks_since_decoder_reset: u64,
    pub replay_chunks: u64,
    pub replay_nonempty_chunks: i64,
    pub replay_inference_ms: i64,
}

pub struct SpeechEndRow {
    pub uptime_ms: i64,
    pub speech_duration_ms: f64,
    pub consecutive_empty: u32,
    pub chunks_since_decoder_reset: u64,
}

pub struct DiagSink {
    conn: Option<Connection>,
    session_id: i64,
    enabled: Arc<AtomicBool>,
    /// Captured at open from the session's starting settings. Hot-reload never
    /// changes `chunk_ms`, and sourcing it live would desync the derived
    /// `audio_ms_since_decoder_reset` columns from the chunks they describe.
    chunk_ms: usize,
}

impl DiagSink {
    /// Opens the DB, applies migrations and inserts the `sessions` row.
    /// A `None` path yields an inert sink that writes nothing.
    pub fn open(
        db_path: Option<&Path>,
        enabled: Arc<AtomicBool>,
        input_rate: usize,
        chunk_size: usize,
        chunk_ms: usize,
        needs_resample: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let conn = init_diag_db(db_path)?;

        let session_id =
            if let Some(conn) = conn.as_ref().filter(|_| enabled.load(Ordering::Relaxed)) {
                conn.execute(
                "INSERT INTO sessions (input_rate, chunk_size, needs_resample) VALUES (?1, ?2, ?3)",
                rusqlite::params![input_rate as i64, chunk_size as i64, needs_resample as i64],
            )?;
                conn.last_insert_rowid()
            } else {
                0
            };

        Ok(Self {
            conn,
            session_id,
            enabled,
            chunk_ms,
        })
    }

    /// The single gate. Reproduces the `db.as_ref().filter(|_| enabled.load(..))`
    /// guard that wrapped every write site.
    fn live(&self) -> Option<&Connection> {
        self.conn
            .as_ref()
            .filter(|_| self.enabled.load(Ordering::Relaxed))
    }

    fn audio_ms(&self, chunks: u64) -> i64 {
        chunks_to_audio_ms(chunks, self.chunk_ms)
    }

    pub fn transcribe(&self, r: &TranscribeRow) {
        let Some(db) = self.live() else { return };
        let _ = db.execute(
            "INSERT INTO events (session_id, uptime_ms, event_type, chunk_num,
                                 inference_ms, drain_samples, drain_audio_ms,
                                 asr_buf_len, text_empty, text_preview, vad_state,
                                 vad_ms, resample_ms, iteration_ms, chunk_source)
                                 VALUES (?1, ?2, 'transcribe', ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            rusqlite::params![
                self.session_id,
                r.uptime_ms,
                r.chunk_num as i64,
                r.inference_ms,
                r.drain_samples as i64,
                r.drain_audio_ms,
                r.asr_buf_len as i64,
                r.is_empty as i64,
                r.preview,
                r.vad_state,
                r.vad_ms,
                r.resample_ms,
                r.iteration_ms,
                r.chunk_source,
            ],
        );
    }

    pub fn asr_error(&self, r: &AsrErrorRow) {
        let Some(db) = self.live() else { return };
        let _ = db.execute(
            "INSERT INTO events (session_id, uptime_ms, event_type, chunk_num,
                                 inference_ms, error_msg, vad_state, chunk_source)
                                 VALUES (?1, ?2, 'asr_error', ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                self.session_id,
                r.uptime_ms,
                r.chunk_num as i64,
                r.inference_ms,
                r.error_msg,
                r.vad_state,
                r.chunk_source,
            ],
        );
    }

    pub fn resample_error(&self, uptime_ms: i64, error_msg: &str) {
        let Some(db) = self.live() else { return };
        let _ = db.execute(
            "INSERT INTO events (session_id, uptime_ms, event_type, error_msg)
                         VALUES (?1, ?2, 'resample_error', ?3)",
            rusqlite::params![self.session_id, uptime_ms, error_msg],
        );
    }

    pub fn vad_error(&self, uptime_ms: i64, error_msg: &str) {
        let Some(db) = self.live() else { return };
        let _ = db.execute(
            "INSERT INTO events (session_id, uptime_ms, event_type, error_msg)
                                 VALUES (?1, ?2, 'vad_error', ?3)",
            rusqlite::params![self.session_id, uptime_ms, error_msg],
        );
    }

    /// Non-fatal or fatal error reported by an ASR backend (e.g. a cloud
    /// provider reconnecting). Reuses the existing `error_msg` column.
    pub fn backend_error(&self, uptime_ms: i64, kind: &str, error_msg: &str) {
        let Some(db) = self.live() else { return };
        let _ = db.execute(
            "INSERT INTO events (session_id, uptime_ms, event_type, error_msg, chunk_source)
                         VALUES (?1, ?2, 'backend_error', ?3, ?4)",
            rusqlite::params![self.session_id, uptime_ms, error_msg, kind],
        );
    }

    pub fn shutdown(&self, uptime_ms: i64, chunk_num: u64) {
        let Some(db) = self.live() else { return };
        let _ = db.execute(
            "INSERT INTO events (session_id, uptime_ms, event_type, chunk_num)
                         VALUES (?1, ?2, 'shutdown', ?3)",
            rusqlite::params![self.session_id, uptime_ms, chunk_num as i64],
        );
    }

    pub fn speech_start(&self, uptime_ms: i64, pre_speech_samples: usize) {
        let Some(db) = self.live() else { return };
        let _ = db.execute(
            "INSERT INTO vad_events (session_id, uptime_ms, event_type, pre_speech_samples)
                                 VALUES (?1, ?2, 'speech_start', ?3)",
            rusqlite::params![self.session_id, uptime_ms, pre_speech_samples as i64],
        );
    }

    pub fn punctuation_reset(&self, r: &PunctuationResetRow) {
        let Some(db) = self.live() else { return };
        let _ = db.execute(
            "INSERT INTO vad_events (
                                        session_id, uptime_ms, event_type, consecutive_empty,
                                        chunks_since_decoder_reset, audio_ms_since_decoder_reset
                                     )
                                     VALUES (?1, ?2, 'punctuation_reset', ?3, ?4, ?5)",
            rusqlite::params![
                self.session_id,
                r.uptime_ms,
                r.consecutive_empty as i64,
                r.chunks_since_decoder_reset as i64,
                self.audio_ms(r.chunks_since_decoder_reset),
            ],
        );
    }

    pub fn mid_speech_reset(&self, r: &MidSpeechResetRow) {
        let Some(db) = self.live() else { return };
        let _ = db.execute(
            "INSERT INTO vad_events (
                                        session_id, uptime_ms, event_type, consecutive_empty,
                                        chunks_since_decoder_reset, audio_ms_since_decoder_reset,
                                        replay_chunks, replay_audio_ms, replay_nonempty_chunks,
                                        replay_inference_ms
                                     )
                                     VALUES (?1, ?2, 'mid_speech_reset', ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                self.session_id,
                r.uptime_ms,
                r.consecutive_empty as i64,
                r.chunks_since_decoder_reset as i64,
                self.audio_ms(r.chunks_since_decoder_reset),
                r.replay_chunks as i64,
                self.audio_ms(r.replay_chunks),
                r.replay_nonempty_chunks,
                r.replay_inference_ms,
            ],
        );
    }

    pub fn speech_end(&self, r: &SpeechEndRow) {
        let Some(db) = self.live() else { return };
        let _ = db.execute(
            "INSERT INTO vad_events (
                                session_id, uptime_ms, event_type, speech_duration_ms,
                                consecutive_empty, chunks_since_decoder_reset,
                                audio_ms_since_decoder_reset
                             )
                             VALUES (?1, ?2, 'speech_end', ?3, ?4, ?5, ?6)",
            rusqlite::params![
                self.session_id,
                r.uptime_ms,
                r.speech_duration_ms,
                r.consecutive_empty as i64,
                r.chunks_since_decoder_reset as i64,
                self.audio_ms(r.chunks_since_decoder_reset),
            ],
        );
    }
}

fn init_diag_db(db_path: Option<&Path>) -> Result<Option<Connection>, Box<dyn std::error::Error>> {
    let Some(db_path) = db_path else {
        return Ok(None);
    };
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
    Ok(Some(conn))
}

fn migrate_add_column(conn: &Connection, sql: &str) {
    if let Err(e) = conn.execute_batch(sql) {
        if !e.to_string().contains("duplicate column name") {
            eprintln!("[diag] Migration failed for `{}`: {}", sql, e);
        }
    }
}

pub fn chunks_to_audio_ms(chunks: u64, chunk_ms: usize) -> i64 {
    chunks.saturating_mul(chunk_ms as u64).min(i64::MAX as u64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_to_audio_ms_uses_chunk_duration() {
        assert_eq!(chunks_to_audio_ms(6, 560), 3360);
    }

    #[test]
    fn disabled_sink_writes_nothing_and_has_no_session() {
        let sink = DiagSink::open(
            None,
            Arc::new(AtomicBool::new(true)),
            48000,
            8960,
            560,
            true,
        )
        .expect("inert sink opens");
        assert_eq!(sink.session_id, 0);
        assert!(sink.live().is_none());
        // Must not panic without a connection.
        sink.shutdown(0, 0);
        sink.speech_start(0, 0);
    }

    #[test]
    fn live_gate_follows_the_shared_flag() {
        let dir = std::env::temp_dir().join("larmindon_diag_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("gate_{}.sqlite", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let flag = Arc::new(AtomicBool::new(true));
        let sink = DiagSink::open(Some(&path), Arc::clone(&flag), 16000, 8960, 560, false).unwrap();
        assert!(sink.session_id > 0);
        assert!(sink.live().is_some());

        // Flipping the shared flag off must suppress writes without closing.
        flag.store(false, Ordering::Relaxed);
        assert!(sink.live().is_none());
        flag.store(true, Ordering::Relaxed);
        assert!(sink.live().is_some());

        drop(sink);
        let _ = std::fs::remove_file(&path);
    }
}
