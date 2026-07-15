use serde_json::{Map, Value};

use crate::server::web::wire::WebError;

pub(crate) struct Speech {
    pub text: String,
    pub model_id: Option<String>,
    pub language_code: Option<String>,
}

pub(crate) fn parse_speech(value: Value) -> Result<Speech, WebError> {
    let object = value.as_object().ok_or_else(|| WebError::Validation("request body must be an object".into()))?;
    let allowed = [
        "text", "model_id", "language_code", "voice_settings", "generation_config", "seed", "use_pvc_as_ivc",
        "apply_text_normalization", "apply_language_text_normalization", "pronunciation_dictionary_locators", "previous_text", "next_text",
        "previous_request_ids", "next_request_ids",
    ];
    if let Some(field) = object.keys().filter(|key| !allowed.contains(&key.as_str())).min() {
        return Err(WebError::Validation(format!("unsupported request field: {field}")));
    }
    let text = match object.get("text") {
        None => return Err(WebError::Issue { location: vec!["body", "text"], message: "Field required", kind: "missing" }),
        Some(Value::String(text)) => text.clone(),
        Some(_) => {
            return Err(WebError::Issue {
                location: vec!["body", "text"],
                message: "Input should be a valid string",
                kind: "string_type",
            });
        }
    };
    validate_nested(object)?;
    Ok(Speech { text, model_id: optional_string(object, "model_id")?, language_code: optional_string(object, "language_code")? })
}

fn validate_nested(object: &Map<String, Value>) -> Result<(), WebError> {
    for field in ["voice_settings", "generation_config"] {
        if let Some(value) = object.get(field) {
            if !value.is_null() && !value.is_object() {
                return Err(WebError::Validation(format!("{field} must be an object")));
            }
        }
    }
    Ok(())
}

fn optional_string(object: &Map<String, Value>, field: &str) -> Result<Option<String>, WebError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if !value.is_empty() => Ok(Some(value.clone())),
        Some(_) => Err(WebError::Validation(format!("{field} must be a non-empty string"))),
    }
}
