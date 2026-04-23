use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const VALID_CHUNK_MS: &[usize] = &[80, 160, 560, 1120];
const VALID_THEMES: &[&str] = &["light", "dark", "system"];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub model_path: String,
    pub chunk_ms: usize,
    pub intra_threads: usize,
    pub inter_threads: usize,
    pub punctuation_reset: bool,
    pub empty_reset_threshold: u32,
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
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model_path: "~/projects/prs-nemotron/".to_string(),
            chunk_ms: 560,
            intra_threads: 2,
            inter_threads: 1,
            punctuation_reset: true,
            empty_reset_threshold: 6,
            font_family: String::new(),
            font_size_px: 0,
            theme_mode: "dark".to_string(),
            vad_threshold_start: 0.5,
            vad_threshold_end: 0.3,
            diagnostics_enabled: false,
            diagnostics_db_path: Self::default_diag_db_path().to_string_lossy().into_owned(),
        }
    }
}

impl Settings {
    /// Returns the config directory: ~/.config/larmindon/
    pub fn config_dir() -> PathBuf {
        if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join(".config").join("larmindon")
        } else {
            // Fallback for non-Unix or missing HOME
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
    pub fn load() -> Self {
        let path = Self::settings_path();
        match std::fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str::<Settings>(&contents) {
                Ok(settings) => {
                    println!("Loaded settings from {}", path.display());
                    settings
                }
                Err(e) => {
                    eprintln!(
                        "Failed to parse settings from {}: {}. Using defaults.",
                        path.display(),
                        e
                    );
                    Self::default()
                }
            },
            Err(_) => {
                println!("No settings file at {}, using defaults.", path.display());
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

    /// Apply environment variable overrides on top of the current settings.
    /// Priority: env var > saved setting > default.
    pub fn with_env_overrides(mut self) -> Self {
        if let Ok(val) = std::env::var("CHUNK_MS") {
            if let Ok(ms) = val.parse::<usize>() {
                if VALID_CHUNK_MS.contains(&ms) {
                    println!("Using CHUNK_MS={ms}ms from environment");
                    self.chunk_ms = ms;
                } else {
                    eprintln!(
                        "Invalid CHUNK_MS={ms}; must be one of {:?}. Keeping saved value {}ms.",
                        VALID_CHUNK_MS, self.chunk_ms
                    );
                }
            } else {
                eprintln!("Could not parse CHUNK_MS={val:?}. Keeping saved value.");
            }
        }

        if let Ok(val) = std::env::var("INTRA_THREADS") {
            match val.parse::<usize>() {
                Ok(n) if n >= 1 => {
                    println!("Using INTRA_THREADS={n} from environment");
                    self.intra_threads = n;
                }
                _ => eprintln!("Invalid INTRA_THREADS={val:?}, keeping saved value."),
            }
        }

        if let Ok(val) = std::env::var("INTER_THREADS") {
            match val.parse::<usize>() {
                Ok(n) if n >= 1 => {
                    println!("Using INTER_THREADS={n} from environment");
                    self.inter_threads = n;
                }
                _ => eprintln!("Invalid INTER_THREADS={val:?}, keeping saved value."),
            }
        }

        if let Ok(val) = std::env::var("PUNCTUATION_RESET") {
            match val.to_lowercase().as_str() {
                "0" | "false" | "no" => {
                    println!(
                        "Punctuation-based decoder reset DISABLED via PUNCTUATION_RESET={val}"
                    );
                    self.punctuation_reset = false;
                }
                "1" | "true" | "yes" => {
                    println!("Punctuation-based decoder reset ENABLED via PUNCTUATION_RESET={val}");
                    self.punctuation_reset = true;
                }
                _ => eprintln!("Unknown PUNCTUATION_RESET={val:?}, keeping saved value."),
            }
        }

        self
    }

    /// Validate that settings values are within acceptable ranges.
    pub fn validate(&self) -> Result<(), String> {
        if !VALID_CHUNK_MS.contains(&self.chunk_ms) {
            return Err(format!(
                "Invalid chunk_ms {}; must be one of {:?}",
                self.chunk_ms, VALID_CHUNK_MS
            ));
        }
        if self.intra_threads < 1 {
            return Err("intra_threads must be at least 1".to_string());
        }
        if self.inter_threads < 1 {
            return Err("inter_threads must be at least 1".to_string());
        }
        if self.empty_reset_threshold < 1 {
            return Err("empty_reset_threshold must be at least 1".to_string());
        }
        if self.model_path.trim().is_empty() {
            return Err("model_path cannot be empty".to_string());
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

        // Warn (but don't error) if model path doesn't exist
        let expanded = expand_tilde(&self.model_path);
        if !expanded.exists() {
            eprintln!("Warning: model path {} does not exist", expanded.display());
        }

        Ok(())
    }
}

/// Expand tilde (~) to home directory in a path
pub fn expand_tilde(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(stripped);
        }
    }
    PathBuf::from(path)
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
    fn validate_rejects_invalid_chunk_ms() {
        let mut s = Settings::default();
        s.chunk_ms = 999;
        assert!(s.validate().is_err());
        assert!(s.validate().unwrap_err().contains("chunk_ms"));
    }

    #[test]
    fn validate_accepts_all_valid_chunk_ms() {
        for &ms in &[80, 160, 560, 1120] {
            let mut s = Settings::default();
            s.chunk_ms = ms;
            assert!(s.validate().is_ok(), "chunk_ms={} should be valid", ms);
        }
    }

    #[test]
    fn validate_rejects_zero_intra_threads() {
        let mut s = Settings::default();
        s.intra_threads = 0;
        assert!(s.validate().is_err());
        assert!(s.validate().unwrap_err().contains("intra_threads"));
    }

    #[test]
    fn validate_rejects_zero_inter_threads() {
        let mut s = Settings::default();
        s.inter_threads = 0;
        assert!(s.validate().is_err());
        assert!(s.validate().unwrap_err().contains("inter_threads"));
    }

    #[test]
    fn validate_rejects_zero_empty_reset_threshold() {
        let mut s = Settings::default();
        s.empty_reset_threshold = 0;
        assert!(s.validate().is_err());
        assert!(s.validate().unwrap_err().contains("empty_reset_threshold"));
    }

    #[test]
    fn validate_rejects_empty_model_path() {
        let mut s = Settings::default();
        s.model_path = "   ".to_string();
        assert!(s.validate().is_err());
        assert!(s.validate().unwrap_err().contains("model_path"));
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
        // Should expand to $HOME/some/path (or unchanged if HOME is unset)
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
        let settings = Settings::default();
        let json = serde_json::to_string(&settings).unwrap();
        let deserialized: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.chunk_ms, settings.chunk_ms);
        assert_eq!(deserialized.model_path, settings.model_path);
        assert_eq!(deserialized.theme_mode, settings.theme_mode);
    }

    #[test]
    fn serde_missing_fields_use_defaults() {
        // Simulate a settings file that only has some fields
        let json = r#"{"chunk_ms": 160}"#;
        let settings: Settings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.chunk_ms, 160);
        // All other fields should be defaults
        assert_eq!(settings.intra_threads, 2);
        assert_eq!(settings.punctuation_reset, true);
    }
}
