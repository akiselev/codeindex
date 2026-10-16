//! Request/response shapes shared by OpenAI-compatible `/embeddings` APIs
//! (OpenAI, OpenRouter, DeepInfra, Together, vLLM/TEI servers, ...).
//! Providers build on these and add their own routing and policy fields.

use serde::Deserialize;
use serde_json::{Value, json};

use super::{Reply, Usage, excerpt, is_transient};

/// The common body: `model`, `input`, float encoding, and the optional
/// `dimensions` / `input_type` extensions.
pub fn request_body(
    model: &str,
    inputs: &[&str],
    dimensions: Option<usize>,
    input_type: Option<&str>,
) -> Value {
    let mut body = json!({
        "model": model,
        "input": inputs,
        "encoding_format": "float",
    });
    if let Some(dimensions) = dimensions {
        body["dimensions"] = json!(dimensions);
    }
    if let Some(input_type) = input_type {
        body["input_type"] = json!(input_type);
    }
    body
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
    #[serde(default)]
    usage: Option<ResponseUsage>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize)]
struct ResponseUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    cost: Option<f64>,
}

/// Classify an OpenAI-style response: HTTP status first, then an `error`
/// object in a 2xx body (some gateways report upstream failures that way),
/// then the `data` list.
pub fn interpret(status: u16, body: &str) -> Reply {
    if !(200..300).contains(&status) {
        let reason = format!("HTTP {status}: {}", excerpt(body));
        return if status == 413 {
            Reply::TooLarge(reason)
        } else if status == 429 {
            Reply::RateLimited(reason)
        } else if is_transient(status) {
            Reply::Retry(reason)
        } else {
            Reply::Fatal(reason)
        };
    }
    let value: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        // A truncated or HTML body from an intermediary; worth one more try.
        Err(_) => return Reply::Retry(format!("non-JSON response: {}", excerpt(body))),
    };
    if let Some(error) = value.get("error") {
        let code = error.get("code").and_then(Value::as_u64).unwrap_or(0) as u16;
        let reason = format!("upstream error: {}", excerpt(&error.to_string()));
        return match code {
            413 => Reply::TooLarge(reason),
            429 => Reply::RateLimited(reason),
            0 => Reply::Retry(reason),
            code if is_transient(code) => Reply::Retry(reason),
            _ => Reply::Fatal(reason),
        };
    }
    match serde_json::from_value::<EmbeddingResponse>(value) {
        Ok(parsed) => Reply::Embeddings {
            data: parsed
                .data
                .into_iter()
                .map(|datum| (datum.index, datum.embedding))
                .collect(),
            usage: parsed.usage.map_or(Usage::default(), |usage| Usage {
                prompt_tokens: usage.prompt_tokens,
                cost: usage.cost,
            }),
        },
        Err(error) => Reply::Fatal(format!("unexpected response ({error}): {}", excerpt(body))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_carries_optional_extensions() {
        let body = request_body("m", &["a", "b"], Some(512), Some("document"));
        assert_eq!(body["input"][1], "b");
        assert_eq!(body["dimensions"], 512);
        assert_eq!(body["input_type"], "document");
        let plain = request_body("m", &["a"], None, None);
        assert!(plain.get("dimensions").is_none() && plain.get("input_type").is_none());
    }

    #[test]
    fn responses_are_classified() {
        let ok = r#"{"data":[{"embedding":[1.0,0.0],"index":1},{"embedding":[0.0,1.0],"index":0}],
                     "usage":{"prompt_tokens":7,"total_tokens":7,"cost":0.0001}}"#;
        match interpret(200, ok) {
            Reply::Embeddings { data, usage } => {
                assert_eq!(data[0].0, 1);
                assert_eq!(usage.prompt_tokens, 7);
                assert_eq!(usage.cost, Some(0.0001));
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(interpret(429, "slow down"), Reply::RateLimited(_)));
        assert!(matches!(interpret(503, "overloaded"), Reply::Retry(_)));
        let upstream_limit = r#"{"error":{"code":429,"message":"rate limited"}}"#;
        assert!(matches!(
            interpret(200, upstream_limit),
            Reply::RateLimited(_)
        ));
        assert!(matches!(interpret(413, "too big"), Reply::TooLarge(_)));
        assert!(matches!(interpret(402, "no credits"), Reply::Fatal(_)));
        let upstream = r#"{"error":{"code":502,"message":"provider down"}}"#;
        assert!(matches!(interpret(200, upstream), Reply::Retry(_)));
        let invalid = r#"{"error":{"code":400,"message":"bad model"}}"#;
        assert!(matches!(interpret(200, invalid), Reply::Fatal(_)));
    }
}
