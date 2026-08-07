//! Soniox realtime WebSocket wire types.
//!
//! Endpoint: `wss://stt-rt.soniox.com/transcribe-websocket`. There is no
//! header-based auth — the key travels in the first message.
//!
//! Response frames have **no type discriminator**: transcripts, errors and the
//! terminal message all share one shape and are told apart by which optional
//! fields are populated. Every field is therefore optional.

use serde::{Deserialize, Serialize};

pub const SONIOX_URL: &str = "wss://stt-rt.soniox.com/transcribe-websocket";

/// Emitted by endpoint detection at the end of a finalized segment.
pub const TOKEN_END: &str = "<end>";
/// The server's acknowledgement of a manual finalize request.
pub const TOKEN_FIN: &str = "<fin>";

/// First text frame, sent immediately after connect.
///
/// Field names are load-bearing and easy to get wrong: it is `num_channels`
/// (not `channels`) and `enable_speaker_diarization` (not `enable_diarization`).
#[derive(Debug, Clone, Serialize)]
pub struct StartRequest {
    pub api_key: String,
    pub model: String,
    pub audio_format: String,
    pub sample_rate: u32,
    pub num_channels: u32,
    pub language_hints: Vec<String>,
    pub language_hints_strict: bool,
    pub enable_speaker_diarization: bool,
    pub enable_language_identification: bool,
    pub enable_endpoint_detection: bool,
}

impl StartRequest {
    pub fn new(
        api_key: String,
        model: String,
        language_hints: Vec<String>,
        diarization: bool,
        endpoint_detection: bool,
    ) -> Self {
        Self {
            api_key,
            model,
            audio_format: "pcm_s16le".to_string(),
            sample_rate: 16_000,
            num_channels: 1,
            language_hints,
            language_hints_strict: false,
            enable_speaker_diarization: diarization,
            enable_language_identification: false,
            enable_endpoint_detection: endpoint_detection,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SonioxToken {
    pub text: String,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub confidence: Option<f64>,
    pub is_final: bool,
    pub speaker: Option<String>,
    pub language: Option<String>,
    pub source_language: Option<String>,
    pub translation_status: Option<String>,
}

impl SonioxToken {
    /// True for the control tokens that must never reach displayed text.
    pub fn is_special(&self) -> bool {
        let t = self.text.trim();
        t == TOKEN_END || t == TOKEN_FIN
    }

    /// Marks the end of a finalized segment; used as a flush signal.
    pub fn is_end_marker(&self) -> bool {
        self.text.trim() == TOKEN_END
    }

    /// Speaker label normalized: whitespace trimmed, empty treated as absent.
    pub fn normalized_speaker(&self) -> Option<String> {
        self.speaker
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SonioxResponse {
    pub tokens: Vec<SonioxToken>,
    pub final_audio_proc_ms: u64,
    pub total_audio_proc_ms: u64,
    pub finished: bool,
    pub error_code: Option<u32>,
    pub error_type: Option<String>,
    pub error_message: Option<String>,
    pub more_info: Option<String>,
    pub request_id: Option<String>,
}

impl SonioxResponse {
    /// Errors are detected by `error_type` being present.
    ///
    /// Not by `error_code`, which defaults to 0 and is absent on some errors,
    /// and not by an empty token list, which is normal during silence.
    pub fn error(&self) -> Option<SonioxError> {
        let error_type = self.error_type.as_deref()?;
        Some(SonioxError {
            error_type: error_type.to_string(),
            message: self
                .error_message
                .clone()
                .unwrap_or_else(|| error_type.to_string()),
            retryable: is_retryable(error_type),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SonioxError {
    pub error_type: String,
    pub message: String,
    pub retryable: bool,
}

/// Transient server-side conditions worth reconnecting through. Everything
/// else — auth, malformed request, unavailable model, expired key, exhausted
/// balance — is permanent and must not be retried.
pub fn is_retryable(error_type: &str) -> bool {
    matches!(
        error_type,
        "request_timeout"
            | "max_duration_reached"
            | "limit_exceeded"
            | "internal_error"
            | "service_unavailable"
    )
}

/// Converts pipeline f32 samples to the signed 16-bit little-endian PCM the
/// service expects.
pub fn f32_to_pcm_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        out.extend_from_slice(&((clamped * 32767.0) as i16).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Captured from a live session; the same three messages the reference
    // client's fixtures use.
    const EVOLUTION: [&str; 3] = [
        r#"{"tokens":[{"text":"How","start_ms":0,"end_ms":300,"confidence":0.81,"is_final":false,"speaker":"2","language":"en"}],"final_audio_proc_ms":0,"total_audio_proc_ms":320}"#,
        r#"{"tokens":[{"text":"How","start_ms":0,"end_ms":300,"confidence":0.98,"is_final":true,"speaker":"1","language":"en"},{"text":" are","start_ms":300,"end_ms":620,"confidence":0.87,"is_final":false,"speaker":"1","language":"en"}],"final_audio_proc_ms":300,"total_audio_proc_ms":620}"#,
        r#"{"tokens":[{"text":" are","start_ms":300,"end_ms":620,"confidence":0.96,"is_final":true,"speaker":"1","language":"en"},{"text":" you","start_ms":620,"end_ms":790,"confidence":0.88,"is_final":false,"speaker":"2","language":"en"}],"final_audio_proc_ms":620,"total_audio_proc_ms":790}"#,
    ];

    const FINISHED: &str =
        r#"{"tokens":[],"final_audio_proc_ms":1560,"total_audio_proc_ms":1680,"finished":true}"#;

    const SERVICE_UNAVAILABLE: &str = r#"{"tokens":[],"error_code":503,"error_type":"service_unavailable","error_message":"Cannot continue request. Please restart the request.","more_info":"https://soniox.com/docs/api-reference/errors#service-unavailable","request_id":"3d37a3bd-5078-47ee-a369-b204e3bbedda"}"#;

    fn parse(s: &str) -> SonioxResponse {
        serde_json::from_str(s).expect("fixture parses")
    }

    #[test]
    fn parses_token_evolution() {
        let first = parse(EVOLUTION[0]);
        assert_eq!(first.tokens.len(), 1);
        assert_eq!(first.tokens[0].text, "How");
        assert!(!first.tokens[0].is_final);
        assert!(first.error().is_none());

        let second = parse(EVOLUTION[1]);
        assert_eq!(second.tokens.len(), 2);
        assert!(second.tokens[0].is_final);
        assert!(!second.tokens[1].is_final);
    }

    #[test]
    fn provisional_speaker_is_revised_on_finalization() {
        // The whole reason speaker is only trusted on final tokens.
        let provisional = parse(EVOLUTION[0]);
        let finalized = parse(EVOLUTION[1]);
        assert_eq!(
            provisional.tokens[0].normalized_speaker().as_deref(),
            Some("2")
        );
        assert_eq!(
            finalized.tokens[0].normalized_speaker().as_deref(),
            Some("1")
        );
    }

    #[test]
    fn terminal_message_is_recognized_and_is_not_an_error() {
        let r = parse(FINISHED);
        assert!(r.finished);
        assert!(r.tokens.is_empty());
        assert!(r.error().is_none());
    }

    #[test]
    fn service_unavailable_is_retryable() {
        let err = parse(SERVICE_UNAVAILABLE).error().expect("error present");
        assert_eq!(err.error_type, "service_unavailable");
        assert!(err.retryable);
    }

    #[test]
    fn unauthenticated_is_fatal() {
        let r = parse(r#"{"tokens":[],"error_type":"unauthenticated","error_message":"bad key"}"#);
        let err = r.error().expect("error present");
        assert!(!err.retryable);
    }

    #[test]
    fn errors_are_detected_without_an_error_code() {
        // error_code defaults to absent; detection must not depend on it.
        let r = parse(r#"{"tokens":[],"error_type":"internal_error"}"#);
        assert!(r.error().is_some());
        assert!(r.error_code.is_none());
    }

    #[test]
    fn empty_token_list_alone_is_not_an_error() {
        let r = parse(r#"{"tokens":[],"total_audio_proc_ms":900}"#);
        assert!(r.error().is_none());
    }

    #[test]
    fn unknown_fields_and_missing_fields_are_tolerated() {
        let r = parse(r#"{"some_new_field":42}"#);
        assert!(r.tokens.is_empty());
        assert!(!r.finished);
    }

    #[test]
    fn special_tokens_are_flagged() {
        let end = SonioxToken {
            text: "<end>".to_string(),
            ..Default::default()
        };
        let fin = SonioxToken {
            text: "<fin>".to_string(),
            ..Default::default()
        };
        let word = SonioxToken {
            text: " hello".to_string(),
            ..Default::default()
        };
        assert!(end.is_special() && end.is_end_marker());
        assert!(fin.is_special() && !fin.is_end_marker());
        assert!(!word.is_special());
    }

    #[test]
    fn blank_speaker_normalizes_to_none() {
        let t = SonioxToken {
            speaker: Some("  ".to_string()),
            ..Default::default()
        };
        assert_eq!(t.normalized_speaker(), None);
        let t2 = SonioxToken {
            speaker: Some(" 3 ".to_string()),
            ..Default::default()
        };
        assert_eq!(t2.normalized_speaker().as_deref(), Some("3"));
    }

    #[test]
    fn pcm_conversion_is_little_endian_and_clamped() {
        assert_eq!(f32_to_pcm_s16le(&[0.0]), vec![0x00, 0x00]);
        assert_eq!(f32_to_pcm_s16le(&[1.0]), vec![0xFF, 0x7F]);
        // Values beyond full scale must not wrap around to the opposite sign.
        assert_eq!(f32_to_pcm_s16le(&[2.0]), vec![0xFF, 0x7F]);
        assert_eq!(f32_to_pcm_s16le(&[-2.0]), vec![0x01, 0x80]);
    }

    #[test]
    fn start_request_uses_the_documented_field_names() {
        let req = StartRequest::new(
            "secret".to_string(),
            "stt-rt-v5".to_string(),
            vec!["en".to_string()],
            true,
            true,
        );
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"num_channels\":1"));
        assert!(json.contains("\"enable_speaker_diarization\":true"));
        assert!(json.contains("\"audio_format\":\"pcm_s16le\""));
        assert!(json.contains("\"sample_rate\":16000"));
        assert!(!json.contains("\"channels\":"));
        assert!(!json.contains("enable_diarization\""));
    }
}
