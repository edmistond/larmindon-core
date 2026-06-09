use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

const VALID_THEMES: &[&str] = &["light", "dark", "system"];

/// Settings file format version. v1 was flat with Nemotron fields at the top
/// level; v2 nests per-engine config under `engines`.
pub const SETTINGS_VERSION: u32 = 2;

/// Engine-specific settings fields that v1 kept at the top level and v2 moves
/// into `engines.nemotron`.
const V1_NEMOTRON_KEYS: &[&str] = &[
    "model_path",
    "chunk_ms",
    "intra_threads",
    "inter_threads",
    "punctuation_reset",
    "empty_reset_threshold",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub version: u32,
    /// Id of the speech engine used for new sessions.
    pub active_engine: String,
    /// Per-engine config blobs, keyed by engine id. Each blob's schema is
    /// owned by the engine crate; entries for engines this build doesn't
    /// include are preserved untouched so the file stays stable across
    /// feature-different builds.
    pub engines: BTreeMap<String, serde_json::Value>,
    /// Font family for transcript display. Empty string = use default.
    pub font_family: String,
    /// Font size in px for transcript display. 0 = use default.
    pub font_size_px: u32,
    /// Theme mode: "light", "dark", or "system"
    pub theme_mode: String,
    /// VAD speech-start threshold (probability to open a speech segment)
    pub vad_threshold_start: f32,
    /// VAD speech-end threshold (probability to close a speech segment)
    pub vad_threshold_end: f32,
    /// Whether to write per-chunk diagnostics to the SQLite DB.
    pub diagnostics_enabled: bool,
    /// Path to the diagnostics SQLite file. Tilde is expanded at use.
    pub diagnostics_db_path: String,
    /// Whether AGC (Automatic Gain Control) is active.
    pub agc_enabled: bool,
    /// AGC target RMS level in dBFS (e.g. -20.0).
    pub agc_target_rms_dbfs: f32,
    /// AGC maximum gain in dB (ceiling to prevent noise-floor amplification).
    pub agc_max_gain_db: f32,
    /// AGC attack time in ms (fast gain reduction when signal gets loud).
    pub agc_attack_ms: f32,
    /// AGC release time in ms (slow gain increase when signal gets quiet).
    pub agc_release_ms: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: SETTINGS_VERSION,
            active_engine: "nemotron".to_string(),
            engines: BTreeMap::new(),
            font_family: String::new(),
            font_size_px: 0,
            theme_mode: "dark".to_string(),
            vad_threshold_start: 0.5,
            vad_threshold_end: 0.3,
            diagnostics_enabled: false,
            diagnostics_db_path: Self::default_diag_db_path().to_string_lossy().into_owned(),
            agc_enabled: false,
            agc_target_rms_dbfs: -20.0,
            agc_max_gain_db: 30.0,
            agc_attack_ms: 20.0,
            agc_release_ms: 400.0,
        }
    }
}

impl Settings {
    /// Returns the config directory: ~/.config/larmindon/
    pub fn config_dir() -> PathBuf {
        if let Some(home) = home_dir() {
            home.join(".config").join("larmindon")
        } else {
            // Fallback for unusual environments with no discoverable home dir.
            PathBuf::from(".config").join("larmindon")
        }
    }

    fn settings_path() -> PathBuf {
        Self::config_dir().join("settings.json")
    }

    /// Default location for the diagnostics SQLite DB: alongside settings.json.
    pub fn default_diag_db_path() -> PathBuf {
        Self::config_dir().join("larmindon_diag.sqlite")
    }

    /// Load settings from disk, falling back to defaults on any error.
    /// A v1 (flat) file is migrated in place: the Nemotron-specific fields
    /// move under `engines.nemotron`, the original is backed up to
    /// `settings.json.v1.bak`, and the migrated form is written back.
    pub fn load() -> Self {
        let path = Self::settings_path();
        let contents = match std::fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(_) => {
                println!("No settings file at {}, using defaults.", path.display());
                return Self::default();
            }
        };

        let mut value: serde_json::Value = match serde_json::from_str(&contents) {
            Ok(value) => value,
            Err(e) => {
                eprintln!(
                    "Failed to parse settings from {}: {}. Using defaults.",
                    path.display(),
                    e
                );
                return Self::default();
            }
        };

        let migrated = migrate_v1_to_v2(&mut value);

        match serde_json::from_value::<Settings>(value) {
            Ok(settings) => {
                println!("Loaded settings from {}", path.display());
                if migrated {
                    let backup = path.with_extension("json.v1.bak");
                    if let Err(e) = std::fs::write(&backup, &contents) {
                        eprintln!(
                            "Failed to back up v1 settings to {}: {}",
                            backup.display(),
                            e
                        );
                    } else {
                        println!(
                            "Migrated v1 settings; original backed up to {}",
                            backup.display()
                        );
                    }
                    if let Err(e) = settings.save() {
                        eprintln!("Failed to save migrated settings: {}", e);
                    }
                }
                settings
            }
            Err(e) => {
                eprintln!(
                    "Failed to interpret settings from {}: {}. Using defaults.",
                    path.display(),
                    e
                );
                Self::default()
            }
        }
    }

    /// Save settings to disk, creating the config directory if needed.
    pub fn save(&self) -> Result<(), String> {
        self.validate()?;

        let dir = Self::config_dir();
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("Failed to create config dir {}: {}", dir.display(), e))?;

        let path = Self::settings_path();
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize settings: {}", e))?;
        std::fs::write(&path, json)
            .map_err(|e| format!("Failed to write settings to {}: {}", path.display(), e))?;

        println!("Settings saved to {}", path.display());
        Ok(())
    }

    /// Validate the shared (engine-independent) fields. Engine config blobs
    /// are validated by their factories via the engine registry.
    pub fn validate(&self) -> Result<(), String> {
        if self.active_engine.trim().is_empty() {
            return Err("active_engine cannot be empty".to_string());
        }
        if !VALID_THEMES.contains(&self.theme_mode.as_str()) {
            return Err(format!(
                "Invalid theme_mode '{}'; must be one of {:?}",
                self.theme_mode, VALID_THEMES
            ));
        }

        if !(0.0..=1.0).contains(&self.vad_threshold_start) {
            return Err(format!(
                "vad_threshold_start must be between 0.0 and 1.0, got {}",
                self.vad_threshold_start
            ));
        }
        if !(0.0..=1.0).contains(&self.vad_threshold_end) {
            return Err(format!(
                "vad_threshold_end must be between 0.0 and 1.0, got {}",
                self.vad_threshold_end
            ));
        }
        if self.vad_threshold_start < self.vad_threshold_end {
            return Err(format!(
                "vad_threshold_start ({}) must be >= vad_threshold_end ({})",
                self.vad_threshold_start, self.vad_threshold_end
            ));
        }

        if self.diagnostics_enabled {
            if self.diagnostics_db_path.trim().is_empty() {
                return Err(
                    "diagnostics_db_path cannot be empty when diagnostics_enabled".to_string(),
                );
            }
            let expanded = expand_tilde(&self.diagnostics_db_path);
            if let Some(parent) = expanded.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        return Err(format!(
                            "Cannot create parent directory {} for diagnostics DB: {}",
                            parent.display(),
                            e
                        ));
                    }
                }
            }
        }

        if !(-60.0..=0.0).contains(&self.agc_target_rms_dbfs) {
            return Err(format!(
                "agc_target_rms_dbfs must be between -60.0 and 0.0, got {}",
                self.agc_target_rms_dbfs
            ));
        }
        if !(0.0..=60.0).contains(&self.agc_max_gain_db) {
            return Err(format!(
                "agc_max_gain_db must be between 0.0 and 60.0, got {}",
                self.agc_max_gain_db
            ));
        }
        if !(1.0..=1000.0).contains(&self.agc_attack_ms) {
            return Err(format!(
                "agc_attack_ms must be between 1.0 and 1000.0, got {}",
                self.agc_attack_ms
            ));
        }
        if !(1.0..=5000.0).contains(&self.agc_release_ms) {
            return Err(format!(
                "agc_release_ms must be between 1.0 and 5000.0, got {}",
                self.agc_release_ms
            ));
        }

        Ok(())
    }

    /// Config blob for the given engine, if one is stored.
    pub fn engine_config(&self, engine_id: &str) -> Option<&serde_json::Value> {
        self.engines.get(engine_id)
    }
}

/// Rewrite a v1 (flat) settings JSON value into the v2 nested shape in place.
/// Returns true if a migration was performed. v2 files pass through untouched.
fn migrate_v1_to_v2(value: &mut serde_json::Value) -> bool {
    let Some(obj) = value.as_object_mut() else {
        return false;
    };
    if obj.contains_key("engines") {
        return false;
    }

    let mut nemotron = serde_json::Map::new();
    for key in V1_NEMOTRON_KEYS {
        if let Some(v) = obj.remove(*key) {
            nemotron.insert(key.to_string(), v);
        }
    }

    let mut engines = serde_json::Map::new();
    engines.insert("nemotron".to_string(), nemotron.into());
    obj.insert("engines".to_string(), engines.into());
    obj.insert("active_engine".to_string(), "nemotron".into());
    obj.insert("version".to_string(), SETTINGS_VERSION.into());
    true
}

/// Expand tilde (~) to home directory in a path
pub fn expand_tilde(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(stripped);
        }
    }
    PathBuf::from(path)
}

fn home_dir() -> Option<PathBuf> {
    if let Some(path) = env_path("HOME") {
        return Some(path);
    }
    if let Some(path) = env_path("USERPROFILE") {
        return Some(path);
    }

    let drive = std::env::var("HOMEDRIVE").ok().filter(|s| !s.is_empty());
    let path = std::env::var("HOMEPATH").ok().filter(|s| !s.is_empty());
    match (drive, path) {
        (Some(drive), Some(path)) => Some(PathBuf::from(format!("{}{}", drive, path))),
        _ => None,
    }
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Convert a chunk duration in milliseconds to samples at 16kHz.
pub fn chunk_ms_to_samples(ms: usize) -> usize {
    16000 * ms / 1000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_are_valid() {
        let settings = Settings::default();
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn validate_rejects_empty_active_engine() {
        let mut s = Settings::default();
        s.active_engine = "  ".to_string();
        assert!(s.validate().is_err());
        assert!(s.validate().unwrap_err().contains("active_engine"));
    }

    #[test]
    fn validate_rejects_invalid_theme_mode() {
        let mut s = Settings::default();
        s.theme_mode = "neon".to_string();
        assert!(s.validate().is_err());
        assert!(s.validate().unwrap_err().contains("theme_mode"));
    }

    #[test]
    fn validate_accepts_all_valid_themes() {
        for theme in &["light", "dark", "system"] {
            let mut s = Settings::default();
            s.theme_mode = theme.to_string();
            assert!(s.validate().is_ok(), "theme '{}' should be valid", theme);
        }
    }

    #[test]
    fn validate_rejects_vad_threshold_out_of_range() {
        let mut s = Settings::default();
        s.vad_threshold_start = 1.5;
        assert!(s.validate().is_err());

        let mut s = Settings::default();
        s.vad_threshold_end = -0.1;
        assert!(s.validate().is_err());
    }

    #[test]
    fn validate_rejects_start_below_end_threshold() {
        let mut s = Settings::default();
        s.vad_threshold_start = 0.2;
        s.vad_threshold_end = 0.5;
        assert!(s.validate().is_err());
        assert!(s.validate().unwrap_err().contains("vad_threshold_start"));
    }

    #[test]
    fn validate_accepts_equal_thresholds() {
        let mut s = Settings::default();
        s.vad_threshold_start = 0.5;
        s.vad_threshold_end = 0.5;
        assert!(s.validate().is_ok());
    }

    #[test]
    fn chunk_ms_to_samples_correct_values() {
        assert_eq!(chunk_ms_to_samples(80), 1280);
        assert_eq!(chunk_ms_to_samples(160), 2560);
        assert_eq!(chunk_ms_to_samples(560), 8960);
        assert_eq!(chunk_ms_to_samples(1120), 17920);
    }

    #[test]
    fn expand_tilde_with_home() {
        let result = expand_tilde("~/some/path");
        // Should expand using the platform's home directory environment.
        assert!(!result.to_string_lossy().starts_with("~/"));
    }

    #[test]
    fn expand_tilde_absolute_path_unchanged() {
        let result = expand_tilde("/absolute/path");
        assert_eq!(result, PathBuf::from("/absolute/path"));
    }

    #[test]
    fn expand_tilde_relative_path_unchanged() {
        let result = expand_tilde("relative/path");
        assert_eq!(result, PathBuf::from("relative/path"));
    }

    #[test]
    fn serde_roundtrip() {
        let mut settings = Settings::default();
        settings
            .engines
            .insert("nemotron".to_string(), serde_json::json!({"chunk_ms": 160}));
        let json = serde_json::to_string(&settings).unwrap();
        let deserialized: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.active_engine, settings.active_engine);
        assert_eq!(deserialized.theme_mode, settings.theme_mode);
        assert_eq!(
            deserialized.engines["nemotron"]["chunk_ms"],
            serde_json::json!(160)
        );
    }

    #[test]
    fn serde_missing_fields_use_defaults() {
        // Simulate a settings file that only has some fields
        let json = r#"{"font_size_px": 18, "engines": {}}"#;
        let settings: Settings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.font_size_px, 18);
        // All other fields should be defaults
        assert_eq!(settings.active_engine, "nemotron");
        assert_eq!(settings.version, SETTINGS_VERSION);
    }

    #[test]
    fn migrate_lifts_v1_engine_fields_into_nemotron_blob() {
        let mut value = serde_json::json!({
            "model_path": "/models/nemotron",
            "chunk_ms": 160,
            "intra_threads": 4,
            "inter_threads": 2,
            "punctuation_reset": false,
            "empty_reset_threshold": 9,
            "theme_mode": "light",
            "font_size_px": 28
        });

        assert!(migrate_v1_to_v2(&mut value));

        let settings: Settings = serde_json::from_value(value).unwrap();
        assert_eq!(settings.version, SETTINGS_VERSION);
        assert_eq!(settings.active_engine, "nemotron");
        assert_eq!(settings.theme_mode, "light");
        assert_eq!(settings.font_size_px, 28);
        let nemotron = &settings.engines["nemotron"];
        assert_eq!(nemotron["model_path"], "/models/nemotron");
        assert_eq!(nemotron["chunk_ms"], 160);
        assert_eq!(nemotron["intra_threads"], 4);
        assert_eq!(nemotron["inter_threads"], 2);
        assert_eq!(nemotron["punctuation_reset"], false);
        assert_eq!(nemotron["empty_reset_threshold"], 9);
    }

    #[test]
    fn migrate_is_idempotent_on_v2_files() {
        let mut value = serde_json::json!({
            "version": 2,
            "active_engine": "april",
            "engines": { "april": { "model_path": "/m.april" } }
        });

        assert!(!migrate_v1_to_v2(&mut value));

        let settings: Settings = serde_json::from_value(value).unwrap();
        assert_eq!(settings.active_engine, "april");
    }

    #[test]
    fn migrate_handles_partial_v1_files() {
        let mut value = serde_json::json!({ "chunk_ms": 1120 });

        assert!(migrate_v1_to_v2(&mut value));

        let settings: Settings = serde_json::from_value(value).unwrap();
        let nemotron = &settings.engines["nemotron"];
        assert_eq!(nemotron["chunk_ms"], 1120);
        // Fields absent in v1 stay absent in the blob; the engine's serde
        // defaults fill them at parse time.
        assert!(nemotron.get("model_path").is_none());
    }

    #[test]
    fn unknown_engine_blobs_survive_roundtrip() {
        let json = r#"{
            "version": 2,
            "active_engine": "nemotron",
            "engines": {
                "nemotron": { "chunk_ms": 560 },
                "some-future-engine": { "api_key": "keep-me" }
            }
        }"#;
        let settings: Settings = serde_json::from_str(json).unwrap();
        let out = serde_json::to_string(&settings).unwrap();
        assert!(out.contains("some-future-engine"));
        assert!(out.contains("keep-me"));
    }
}
