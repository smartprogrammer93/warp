//! Thin HTTP client around llama-server's OpenAI-compatible
//! `/v1/chat/completions` endpoint, returning a stream of raw SSE `data:`
//! payloads (still serialized JSON; deserialization happens in
//! `translate_response.rs`).

use crate::error::LlamaBackendError;
use crate::openai_types::ChatCompletionRequest;
use async_stream::try_stream;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};

/// Stream of raw SSE payload strings (JSON, no `data: ` prefix or trailing
/// `\n\n`). The terminal `[DONE]` marker is consumed and ends the stream
/// without yielding an item.
pub type RawSseStream =
    BoxStream<'static, Result<String, LlamaBackendError>>;

pub struct LlamaClient {
    http: reqwest::Client,
    url: String,
    api_key: Option<String>,
}

impl LlamaClient {
    pub fn new(url: String, api_key: Option<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            url: url.trim_end_matches('/').to_string(),
            api_key,
        }
    }

    /// Open a streaming chat completion against the configured server.
    ///
    /// The returned stream yields the inner JSON of each `data: {...}` SSE
    /// frame; the trailing `data: [DONE]` is dropped silently *and the
    /// underlying connection is closed* (otherwise `reqwest_eventsource`
    /// treats the SSE as long-lived and waits for more frames forever).
    pub fn stream_completion(
        &self,
        req: &ChatCompletionRequest,
    ) -> Result<RawSseStream, LlamaBackendError> {
        let url = format!("{}/v1/chat/completions", self.url);
        let mut builder = self.http.post(&url).json(req);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        let mut es = EventSource::new(builder)
            .map_err(|e| LlamaBackendError::Sse(format!("could not open SSE: {e}")))?;

        let stream = try_stream! {
            while let Some(ev) = es.next().await {
                match ev {
                    Ok(Event::Open) => continue,
                    Ok(Event::Message(msg)) => {
                        if msg.data == "[DONE]" {
                            es.close();
                            break;
                        }
                        yield msg.data;
                    }
                    Err(reqwest_eventsource::Error::StreamEnded) => break,
                    Err(reqwest_eventsource::Error::InvalidStatusCode(status, response)) => {
                        let status_code = status.as_u16();
                        let body = match response.text().await {
                            Ok(t) => t,
                            Err(_) => "<could not read body>".to_string(),
                        };
                        if status_code == 401 || status_code == 403 {
                            Err(LlamaBackendError::InvalidApiKey(body))?;
                        } else {
                            Err(LlamaBackendError::HttpStatus {
                                status: status_code,
                                body,
                            })?;
                        }
                        unreachable!();
                    }
                    Err(other) => {
                        Err(LlamaBackendError::Sse(other.to_string()))?;
                        unreachable!();
                    }
                }
            }
        };
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_types::{OpenAIMessage, OpenAIRole};
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Reqwest is configured workspace-wide with `rustls-tls-no-provider`,
    /// which requires a crypto provider to be installed before any reqwest
    /// `Client` is constructed (even for plain HTTP). The app does this at
    /// startup; for tests we do it ourselves once.
    fn ensure_crypto_provider() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        });
    }

    fn dummy_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test".into(),
            messages: vec![OpenAIMessage {
                role: OpenAIRole::User,
                content: Some("ping".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: vec![],
            stream: true,
            max_tokens: None,
            temperature: None,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streams_two_chunks_then_done() {
        ensure_crypto_provider();
        let server = MockServer::start().await;
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
                    data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(body.as_bytes().to_vec(), "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = LlamaClient::new(server.uri(), None);
        let mut stream = client.stream_completion(&dummy_request()).unwrap();
        let mut payloads = vec![];
        while let Some(item) = stream.next().await {
            payloads.push(item.unwrap());
        }
        assert_eq!(payloads.len(), 2);
        assert!(payloads[0].contains("hello"));
        assert!(payloads[1].contains(" world"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn forwards_api_key_in_authorization_header() {
        ensure_crypto_provider();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer secret-x"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(b"data: [DONE]\n\n".to_vec(), "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = LlamaClient::new(server.uri(), Some("secret-x".into()));
        let mut stream = client.stream_completion(&dummy_request()).unwrap();
        // Drain the stream; if the auth header didn't match, wiremock would
        // 404 and we'd get a HttpStatus error instead.
        while let Some(item) = stream.next().await {
            item.unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn http_401_yields_invalid_api_key_error() {
        ensure_crypto_provider();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(401)
                    .insert_header("content-type", "application/json")
                    .set_body_string(r#"{"error":"unauthorized"}"#),
            )
            .mount(&server)
            .await;

        let client = LlamaClient::new(server.uri(), None);
        let mut stream = client.stream_completion(&dummy_request()).unwrap();
        let first = stream.next().await.expect("error item");
        match first {
            Err(LlamaBackendError::InvalidApiKey(body)) => {
                assert!(body.contains("unauthorized"));
            }
            other => panic!("expected InvalidApiKey, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn http_500_yields_status_error() {
        ensure_crypto_provider();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(500)
                    .insert_header("content-type", "text/plain")
                    .set_body_string("internal boom"),
            )
            .mount(&server)
            .await;

        let client = LlamaClient::new(server.uri(), None);
        let mut stream = client.stream_completion(&dummy_request()).unwrap();
        let first = stream.next().await.expect("error item");
        match first {
            Err(LlamaBackendError::HttpStatus { status, body }) => {
                assert_eq!(status, 500);
                assert!(body.contains("boom"));
            }
            other => panic!("expected HttpStatus, got {other:?}"),
        }
    }
}
