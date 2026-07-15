use serde_json::{Map, Value, json};

use crate::StreamParams;
use crate::server::web::wire::WebError;

const DEFAULT_SCHEDULE: [usize; 4] = [120, 160, 250, 290];

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct VoiceSettings {
    speed: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    stability: Option<f64>,
    similarity_boost: Option<f64>,
    style: Option<f64>,
    use_speaker_boost: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Speech {
    pub text: String,
    pub model_id: Option<String>,
    pub language_code: Option<String>,
    pub(crate) voice_settings: VoiceSettings,
    pub(crate) chunk_length_schedule: Vec<usize>,
    pub(crate) seed: Option<u64>,
    pub(crate) apply_text_normalization: Option<String>,
}

impl Speech {
    pub(crate) fn stream_params(&self, alignment_type: crate::AlignmentType) -> StreamParams {
        let mut model = Map::new();
        for (field, value) in [
            ("speed", self.voice_settings.speed),
            ("style_interpolation_factor", self.voice_settings.stability),
            ("embedding_scale", self.voice_settings.style.map(|style| 1.0 + style)),
            ("alpha", self.voice_settings.alpha.or_else(|| self.voice_settings.similarity_boost.map(|value| 1.0 - value))),
            ("beta", self.voice_settings.beta.or_else(|| self.voice_settings.similarity_boost.map(|value| 1.0 - value))),
        ] {
            if let Some(value) = value {
                model.insert(field.into(), json!(value));
            }
        }
        if let Some(seed) = self.seed {
            model.insert("seed".into(), json!(seed));
        }
        if let Some(language) = &self.language_code {
            model.insert("language".into(), json!(language));
        }
        if let Some(normalization) = &self.apply_text_normalization {
            model.insert("apply_text_normalization".into(), json!(normalization));
        }
        StreamParams {
            chunk_length_schedule: self.chunk_length_schedule.clone(),
            alignment_type,
            model: Value::Object(model),
            ..StreamParams::default()
        }
    }
}

pub(crate) fn parse_speech(value: Value) -> Result<Speech, WebError> {
    let object = value.as_object().ok_or_else(|| WebError::Validation("request body must be an object".into()))?;
    let allowed = [
        "text",
        "model_id",
        "language_code",
        "voice_settings",
        "generation_config",
        "seed",
        "use_pvc_as_ivc",
        "apply_text_normalization",
        "apply_language_text_normalization",
        "pronunciation_dictionary_locators",
        "previous_text",
        "next_text",
        "previous_request_ids",
        "next_request_ids",
    ];
    reject_unknown(object, &allowed, "request")?;
    let text = match object.get("text") {
        None => return Err(WebError::Issue { location: vec!["body", "text"], message: "Field required", kind: "missing" }),
        Some(Value::String(text)) => text.clone(),
        Some(_) => {
            return Err(WebError::Issue { location: vec!["body", "text"], message: "Input should be a valid string", kind: "string_type" });
        }
    };
    validate_optional_bool(object, "use_pvc_as_ivc")?;
    validate_optional_bool(object, "apply_language_text_normalization")?;
    validate_string_sequence(object, "previous_request_ids")?;
    validate_string_sequence(object, "next_request_ids")?;
    validate_optional_text(object, "previous_text")?;
    validate_optional_text(object, "next_text")?;
    validate_dictionaries(object)?;
    Ok(Speech {
        text,
        model_id: optional_string(object, "model_id")?,
        language_code: optional_string(object, "language_code")?,
        voice_settings: parse_voice(optional_object(object, "voice_settings")?)?,
        chunk_length_schedule: parse_schedule(optional_object(object, "generation_config")?)?,
        seed: bounded_integer(object, "seed", 0, u32::MAX as u64)?,
        apply_text_normalization: optional_normalization(object, "apply_text_normalization")?,
    })
}

fn parse_voice(object: &Map<String, Value>) -> Result<VoiceSettings, WebError> {
    reject_unknown(object, &["speed", "alpha", "beta", "stability", "similarity_boost", "style", "use_speaker_boost"], "voice_settings")?;
    Ok(VoiceSettings {
        speed: bounded_float(object, "speed", 0.7, 1.2)?,
        alpha: bounded_float(object, "alpha", 0.0, 1.0)?,
        beta: bounded_float(object, "beta", 0.0, 1.0)?,
        stability: bounded_float(object, "stability", 0.0, 1.0)?,
        similarity_boost: bounded_float(object, "similarity_boost", 0.0, 1.0)?,
        style: bounded_float(object, "style", 0.0, 1.0)?,
        use_speaker_boost: optional_bool(object, "use_speaker_boost")?,
    })
}

fn parse_schedule(object: &Map<String, Value>) -> Result<Vec<usize>, WebError> {
    reject_unknown(object, &["chunk_length_schedule"], "generation_config")?;
    let Some(value) = object.get("chunk_length_schedule") else { return Ok(DEFAULT_SCHEDULE.to_vec()) };
    let array = value
        .as_array()
        .filter(|array| !array.is_empty())
        .ok_or_else(|| WebError::Validation("chunk_length_schedule must be a non-empty array".into()))?;
    array
        .iter()
        .map(|item| {
            let value = item.as_u64().ok_or_else(|| WebError::Validation("chunk_length_schedule values must be integers".into()))?;
            if !(50..=500).contains(&value) {
                return Err(WebError::Validation("chunk_length_schedule values must be 50 through 500".into()));
            }
            Ok(value as usize)
        })
        .collect()
}

fn validate_dictionaries(object: &Map<String, Value>) -> Result<(), WebError> {
    let Some(value) = object.get("pronunciation_dictionary_locators").filter(|value| !value.is_null()) else { return Ok(()) };
    let array = value.as_array().ok_or_else(|| WebError::Validation("pronunciation_dictionary_locators must be an array".into()))?;
    if array.len() > 3 {
        return Err(WebError::Validation("pronunciation_dictionary_locators accepts a maximum of 3 values".into()));
    }
    for locator in array {
        let locator =
            locator.as_object().ok_or_else(|| WebError::Validation("pronunciation_dictionary_locators values must be objects".into()))?;
        reject_unknown(locator, &["pronunciation_dictionary_id", "version_id"], "pronunciation_dictionary_locators")?;
        if optional_string(locator, "pronunciation_dictionary_id")?.is_none() {
            return Err(WebError::Validation("pronunciation_dictionary_id is required".into()));
        }
        optional_string(locator, "version_id")?;
    }
    Ok(())
}

fn reject_unknown(object: &Map<String, Value>, allowed: &[&str], scope: &str) -> Result<(), WebError> {
    if let Some(field) = object.keys().filter(|key| !allowed.contains(&key.as_str())).min() {
        return Err(WebError::Validation(format!("unsupported {scope} field: {field}")));
    }
    Ok(())
}

fn optional_object<'a>(object: &'a Map<String, Value>, field: &str) -> Result<&'a Map<String, Value>, WebError> {
    static EMPTY: std::sync::LazyLock<Map<String, Value>> = std::sync::LazyLock::new(Map::new);
    match object.get(field) {
        None | Some(Value::Null) => Ok(&EMPTY),
        Some(Value::Object(value)) => Ok(value),
        Some(_) => Err(WebError::Validation(format!("{field} must be an object"))),
    }
}

fn optional_string(object: &Map<String, Value>, field: &str) -> Result<Option<String>, WebError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if !value.is_empty() => Ok(Some(value.clone())),
        Some(_) => Err(WebError::Validation(format!("{field} must be a string"))),
    }
}

fn optional_bool(object: &Map<String, Value>, field: &str) -> Result<Option<bool>, WebError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(WebError::Validation(format!("{field} must be boolean"))),
    }
}

fn validate_optional_bool(object: &Map<String, Value>, field: &str) -> Result<(), WebError> {
    optional_bool(object, field).map(|_| ())
}

fn bounded_float(object: &Map<String, Value>, field: &str, minimum: f64, maximum: f64) -> Result<Option<f64>, WebError> {
    let Some(value) = object.get(field).filter(|value| !value.is_null()) else { return Ok(None) };
    let value = value.as_f64().ok_or_else(|| WebError::Validation(format!("{field} must be numeric")))?;
    if !(minimum..=maximum).contains(&value) {
        return Err(WebError::Validation(format!("{field} must be between {minimum} and {maximum}")));
    }
    Ok(Some(value))
}

fn bounded_integer(object: &Map<String, Value>, field: &str, minimum: u64, maximum: u64) -> Result<Option<u64>, WebError> {
    let Some(value) = object.get(field).filter(|value| !value.is_null()) else { return Ok(None) };
    let value = value.as_u64().ok_or_else(|| WebError::Validation(format!("{field} must be an integer")))?;
    if !(minimum..=maximum).contains(&value) {
        return Err(WebError::Validation(format!("{field} must be between {minimum} and {maximum}")));
    }
    Ok(Some(value))
}

fn validate_string_sequence(object: &Map<String, Value>, field: &str) -> Result<(), WebError> {
    let Some(value) = object.get(field).filter(|value| !value.is_null()) else { return Ok(()) };
    let array = value
        .as_array()
        .filter(|array| array.iter().all(Value::is_string))
        .ok_or_else(|| WebError::Validation(format!("{field} must be an array of strings")))?;
    if array.len() > 3 {
        return Err(WebError::Validation(format!("{field} accepts a maximum of 3 values")));
    }
    Ok(())
}

fn validate_optional_text(object: &Map<String, Value>, field: &str) -> Result<(), WebError> {
    match object.get(field) {
        None | Some(Value::Null) | Some(Value::String(_)) => Ok(()),
        Some(_) => Err(WebError::Validation(format!("{field} must be a string"))),
    }
}

fn optional_normalization(object: &Map<String, Value>, field: &str) -> Result<Option<String>, WebError> {
    let value = optional_string(object, field)?;
    if value.as_deref().is_some_and(|value| !matches!(value, "auto" | "on" | "off")) {
        return Err(WebError::Validation(format!("{field} must be auto, on, or off")));
    }
    Ok(value)
}
