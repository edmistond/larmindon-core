//! Engine discovery and construction.
//!
//! Engine crates provide an [`EngineFactory`]; the application shell registers
//! the factories it was compiled with into an [`EngineRegistry`]. The
//! [`EngineDescriptor`] carries enough metadata (including a typed config
//! field list) for the frontend to render a settings UI for any engine without
//! hardcoding it.

use std::sync::Arc;

use super::{EngineError, SpeechEngine};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    Local,
    Cloud,
}

#[derive(Clone, serde::Serialize)]
pub struct EngineDescriptor {
    pub id: &'static str,
    /// Human-readable name shown in the engine selector.
    pub name: &'static str,
    pub kind: EngineKind,
    /// Whether this engine emits transient (revisable) segments. Engines with
    /// `false` only ever produce finalized segments.
    pub emits_partials: bool,
    pub config_fields: Vec<ConfigField>,
}

#[derive(Clone, serde::Serialize)]
pub struct ConfigField {
    /// Key in the engine's config blob.
    pub key: &'static str,
    pub label: &'static str,
    #[serde(flatten)]
    pub field: FieldType,
    pub default: serde_json::Value,
    /// Environment variable that overrides this field at startup.
    pub env_var: Option<&'static str>,
    pub help: Option<&'static str>,
}

#[derive(Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FieldType {
    Bool,
    Int {
        min: i64,
        max: i64,
    },
    Float {
        min: f64,
        max: f64,
        step: f64,
    },
    Enum {
        options: Vec<EnumOption>,
    },
    Path {
        directory: bool,
    },
    Text,
    /// Sensitive value (API key); rendered as a password input.
    Secret,
}

#[derive(Clone, serde::Serialize)]
pub struct EnumOption {
    pub value: serde_json::Value,
    pub label: String,
}

pub trait EngineFactory: Send + Sync {
    fn descriptor(&self) -> EngineDescriptor;

    fn default_config(&self) -> serde_json::Value;

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), String>;

    /// Hash of the config fields that require constructing a fresh engine
    /// (model path, thread counts) — NOT fields a reused instance can apply at
    /// `begin_session` or hot-reload. Drives the cross-session engine cache.
    fn cache_key(&self, config: &serde_json::Value) -> u64;

    /// Construct the engine. Must be cheap; heavy resources (models, network
    /// connections) load in `begin_session` on the processing thread.
    fn create(&self, config: &serde_json::Value) -> Result<Box<dyn SpeechEngine>, EngineError>;
}

#[derive(Default)]
pub struct EngineRegistry {
    factories: Vec<Arc<dyn EngineFactory>>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, factory: Arc<dyn EngineFactory>) {
        self.factories.push(factory);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn EngineFactory>> {
        self.factories
            .iter()
            .find(|f| f.descriptor().id == id)
            .cloned()
    }

    pub fn descriptors(&self) -> Vec<EngineDescriptor> {
        self.factories.iter().map(|f| f.descriptor()).collect()
    }

    /// The first registered engine, used when no explicit selection exists.
    pub fn default_engine_id(&self) -> Option<&'static str> {
        self.factories.first().map(|f| f.descriptor().id)
    }

    /// Apply environment-variable overrides onto per-engine config blobs,
    /// driven by the `env_var` declarations in each engine's descriptor.
    /// Priority: env var > saved setting > default.
    pub fn apply_env_overrides(
        &self,
        engines: &mut std::collections::BTreeMap<String, serde_json::Value>,
    ) {
        for descriptor in self.descriptors() {
            for field in &descriptor.config_fields {
                let Some(env_var) = field.env_var else {
                    continue;
                };
                let Ok(raw) = std::env::var(env_var) else {
                    continue;
                };
                match parse_env_value(&raw, &field.field) {
                    Ok(value) => {
                        println!(
                            "Using {}={} from environment for engine '{}' ({})",
                            env_var, raw, descriptor.id, field.key
                        );
                        let blob = engines
                            .entry(descriptor.id.to_string())
                            .or_insert_with(|| serde_json::json!({}));
                        if let Some(obj) = blob.as_object_mut() {
                            obj.insert(field.key.to_string(), value);
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "Ignoring {}={:?}: {}. Keeping saved value.",
                            env_var, raw, e
                        )
                    }
                }
            }
        }
    }
}

/// Parse an environment-variable string into a config value matching the
/// field's declared type, enforcing the same constraints the UI would.
fn parse_env_value(raw: &str, field: &FieldType) -> Result<serde_json::Value, String> {
    match field {
        FieldType::Bool => match raw.to_lowercase().as_str() {
            "1" | "true" | "yes" => Ok(true.into()),
            "0" | "false" | "no" => Ok(false.into()),
            _ => Err("expected a boolean (0/1/true/false/yes/no)".to_string()),
        },
        FieldType::Int { min, max } => {
            let n: i64 = raw.parse().map_err(|_| "expected an integer".to_string())?;
            if n < *min || n > *max {
                return Err(format!("must be between {} and {}", min, max));
            }
            Ok(n.into())
        }
        FieldType::Float { min, max, .. } => {
            let n: f64 = raw.parse().map_err(|_| "expected a number".to_string())?;
            if n < *min || n > *max {
                return Err(format!("must be between {} and {}", min, max));
            }
            Ok(n.into())
        }
        FieldType::Enum { options } => {
            let candidate: serde_json::Value = match raw.parse::<i64>() {
                Ok(n) => n.into(),
                Err(_) => raw.into(),
            };
            if options.iter().any(|o| o.value == candidate) {
                Ok(candidate)
            } else {
                let valid: Vec<String> = options.iter().map(|o| o.value.to_string()).collect();
                Err(format!("must be one of {}", valid.join(", ")))
            }
        }
        FieldType::Path { .. } | FieldType::Text | FieldType::Secret => Ok(raw.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_env_value_bool_accepts_legacy_forms() {
        assert_eq!(parse_env_value("yes", &FieldType::Bool).unwrap(), true);
        assert_eq!(parse_env_value("0", &FieldType::Bool).unwrap(), false);
        assert!(parse_env_value("maybe", &FieldType::Bool).is_err());
    }

    #[test]
    fn parse_env_value_int_enforces_bounds() {
        let field = FieldType::Int { min: 1, max: 32 };
        assert_eq!(parse_env_value("4", &field).unwrap(), 4);
        assert!(parse_env_value("0", &field).is_err());
        assert!(parse_env_value("x", &field).is_err());
    }

    #[test]
    fn parse_env_value_enum_matches_numeric_options() {
        let field = FieldType::Enum {
            options: vec![
                EnumOption {
                    value: 80.into(),
                    label: "80 ms".to_string(),
                },
                EnumOption {
                    value: 560.into(),
                    label: "560 ms".to_string(),
                },
            ],
        };
        assert_eq!(parse_env_value("560", &field).unwrap(), 560);
        assert!(parse_env_value("999", &field).is_err());
    }
}
