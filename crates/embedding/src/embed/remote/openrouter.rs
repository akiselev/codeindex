//! OpenRouter: an aggregator in front of many upstream embedding providers.
//! Two things make it different from a direct API, and both are expressed in
//! the request's `provider` object:
//! - **Routing.** The same model id can be served by several upstreams with
//!   different numerics (bf16 vs fp8). `providers` pins the order with
//!   fallbacks disabled; `quantizations` filters endpoints. Both enter the
//!   model contract.
//! - **Retention.** Upstreams have their own logging and training terms.
//!   [`DataPolicy::Zdr`] sends `zdr: true` and `data_collection: "deny"`, so
//!   requests only reach zero-data-retention endpoints and fail rather than
//!   fall back to one that retains.

use anyhow::Result;
use serde_json::{Value, json};

use super::{ProviderInit, RemoteProvider, RemoteProviderSpec, Reply, openai_compat};
use crate::config::DataPolicy;
use crate::embed::EmbeddingRole;

pub const SPEC: RemoteProviderSpec = RemoteProviderSpec {
    id: "openrouter",
    default_base_url: "https://openrouter.ai/api/v1",
    default_api_key_env: "OPENROUTER_API_KEY",
    build,
};

fn build(init: ProviderInit<'_>) -> Result<Box<dyn RemoteProvider>> {
    let config = init.config;
    Ok(Box::new(OpenRouter {
        model: init.model.to_string(),
        base_url: init.base_url,
        api_key: init.api_key,
        data_policy: init.data_policy,
        providers: config.providers.clone(),
        quantizations: config.quantizations.clone(),
        dimensions: config.request_dimensions.then_some(config.dimensions),
        document_input_type: config.document_input_type.clone(),
        query_input_type: config.query_input_type.clone(),
    }))
}

pub struct OpenRouter {
    model: String,
    base_url: String,
    api_key: String,
    data_policy: DataPolicy,
    providers: Vec<String>,
    quantizations: Vec<String>,
    dimensions: Option<usize>,
    document_input_type: Option<String>,
    query_input_type: Option<String>,
}

impl RemoteProvider for OpenRouter {
    fn id(&self) -> &'static str {
        SPEC.id
    }

    fn endpoint(&self) -> String {
        format!("{}/embeddings", self.base_url)
    }

    fn headers(&self) -> Vec<(&'static str, String)> {
        vec![
            ("authorization", format!("Bearer {}", self.api_key)),
            ("x-openrouter-title", "codeindex".into()),
        ]
    }

    fn request_body(&self, role: EmbeddingRole, inputs: &[&str]) -> Value {
        let input_type = match role {
            EmbeddingRole::Document => self.document_input_type.as_deref(),
            EmbeddingRole::Query => self.query_input_type.as_deref(),
        };
        let mut body =
            openai_compat::request_body(&self.model, inputs, self.dimensions, input_type);
        let mut provider = serde_json::Map::new();
        match self.data_policy {
            DataPolicy::Zdr => {
                provider.insert("zdr".into(), json!(true));
                provider.insert("data_collection".into(), json!("deny"));
            }
            DataPolicy::NoTraining => {
                provider.insert("data_collection".into(), json!("deny"));
            }
            DataPolicy::ProviderDefault => {}
        }
        if !self.providers.is_empty() {
            provider.insert("order".into(), json!(self.providers));
            provider.insert("allow_fallbacks".into(), json!(false));
        }
        if !self.quantizations.is_empty() {
            provider.insert("quantizations".into(), json!(self.quantizations));
        }
        // An upstream that silently drops `dimensions`/`input_type` would
        // produce different vectors; refuse to route to it.
        if self.dimensions.is_some() || input_type.is_some() {
            provider.insert("require_parameters".into(), json!(true));
        }
        if !provider.is_empty() {
            body["provider"] = Value::Object(provider);
        }
        body
    }

    fn interpret(&self, status: u16, body: &str) -> Reply {
        openai_compat::interpret(status, body)
    }

    fn revision(&self) -> Option<String> {
        let mut parts = Vec::new();
        if !self.providers.is_empty() {
            parts.push(format!("providers={}", self.providers.join(",")));
        }
        if !self.quantizations.is_empty() {
            parts.push(format!("quantizations={}", self.quantizations.join(",")));
        }
        if let Some(input_type) = &self.document_input_type {
            parts.push(format!("input_type={input_type}"));
        }
        (!parts.is_empty()).then(|| parts.join(";"))
    }

    fn routing(&self) -> String {
        let route = if self.providers.is_empty() {
            "auto".to_string()
        } else {
            self.providers.join(",")
        };
        format!("{route} (data_policy={})", self.data_policy.as_str())
    }

    fn disclosure(&self) -> String {
        let route = if self.providers.is_empty() {
            "any upstream OpenRouter selects".to_string()
        } else {
            format!("upstream {}", self.providers.join(" > "))
        };
        let terms = match self.data_policy {
            DataPolicy::Zdr => "zero-data-retention endpoints only",
            DataPolicy::NoTraining => {
                "endpoints that do not train on inputs (operational logging may still apply)"
            }
            DataPolicy::ProviderDefault => {
                "the upstream's default retention terms (may log or train on inputs)"
            }
        };
        let mut line = format!(
            "note: embedding sends code bodies to OpenRouter ({route}) for {}; data policy {}: \
             {terms}",
            self.model,
            self.data_policy.as_str()
        );
        if self.providers.is_empty() {
            line.push_str(
                "\nwarning: no upstream pinned (`providers`), so vectors may come from \
                 different upstreams whose numerics disagree; pin one for a reproducible \
                 database",
            );
        }
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RemoteModelConfig;

    fn provider(policy: DataPolicy, config: RemoteModelConfig) -> Box<dyn RemoteProvider> {
        build(ProviderInit {
            model: "qwen/qwen3-embedding-8b",
            base_url: SPEC.default_base_url.into(),
            api_key: "test".into(),
            data_policy: policy,
            config: &config,
        })
        .unwrap()
    }

    #[test]
    fn data_policy_maps_to_provider_preferences() {
        let zdr = provider(DataPolicy::Zdr, RemoteModelConfig::default());
        let body = zdr.request_body(EmbeddingRole::Document, &["fn a() {}"]);
        assert_eq!(body["provider"]["zdr"], true);
        assert_eq!(body["provider"]["data_collection"], "deny");
        assert!(zdr.disclosure().contains("zero-data-retention"));
        assert!(zdr.disclosure().contains("no upstream pinned"));

        let no_training = provider(DataPolicy::NoTraining, RemoteModelConfig::default());
        let body = no_training.request_body(EmbeddingRole::Document, &["x"]);
        assert!(body["provider"].get("zdr").is_none());
        assert_eq!(body["provider"]["data_collection"], "deny");

        let open = provider(DataPolicy::ProviderDefault, RemoteModelConfig::default());
        let body = open.request_body(EmbeddingRole::Document, &["x"]);
        assert!(body.get("provider").is_none());
        assert!(open.disclosure().contains("may log or train"));
    }

    #[test]
    fn routing_pins_and_parameters() {
        let pinned = provider(
            DataPolicy::Zdr,
            RemoteModelConfig {
                dimensions: 1024,
                request_dimensions: true,
                providers: vec!["deepinfra".into()],
                quantizations: vec!["bf16".into()],
                document_input_type: Some("document".into()),
                query_input_type: Some("query".into()),
                ..RemoteModelConfig::default()
            },
        );
        let body = pinned.request_body(EmbeddingRole::Query, &["find parser"]);
        assert_eq!(body["model"], "qwen/qwen3-embedding-8b");
        assert_eq!(body["dimensions"], 1024);
        assert_eq!(body["input_type"], "query");
        assert_eq!(body["provider"]["order"][0], "deepinfra");
        assert_eq!(body["provider"]["allow_fallbacks"], false);
        assert_eq!(body["provider"]["quantizations"][0], "bf16");
        assert_eq!(body["provider"]["require_parameters"], true);
        assert_eq!(
            pinned.revision().as_deref(),
            Some("providers=deepinfra;quantizations=bf16;input_type=document")
        );
        assert_eq!(pinned.endpoint(), "https://openrouter.ai/api/v1/embeddings");

        let auto = provider(DataPolicy::Zdr, RemoteModelConfig::default());
        assert_eq!(auto.revision(), None);
        assert_eq!(auto.routing(), "auto (data_policy=zdr)");
    }
}
