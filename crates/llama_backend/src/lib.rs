//! Local-LLM backend for Warp.
//!
//! Replaces the network call to `https://app.warp.dev/ai/multi-agent` with
//! a direct call to a local OpenAI-compatible server (e.g. llama-server),
//! when the env var `WARP_LLAMA_URL` is set.
//!
//! Integration point is `app/src/server/server_api.rs::generate_multi_agent_output`,
//! which calls [`dispatch_if_enabled`] before doing its real HTTP+SSE work; if
//! the function returns `Some`, the upstream caller returns that stream
//! verbatim.
//!
//! Per-turn flow on the local backend:
//!
//! 1. Resolve the conversation id (mint a fresh UUID on first turn).
//! 2. Update the in-process [`ConversationStore`] with the new turn's
//!    contributions (user message, tool results).
//! 3. Translate the proto request to OpenAI format ([`translate_request`])
//!    and stream a chat completion from llama-server ([`llama_client`]).
//! 4. Translate the OpenAI SSE chunks back into the transactional
//!    `ResponseEvent` sequence Warp's UI consumes
//!    ([`translate_response`]).

#![deny(unused_must_use)]

use futures::stream::BoxStream;
use futures::StreamExt;
use std::sync::OnceLock;
use warp_multi_agent_api::{Request, ResponseEvent};

pub mod config;
pub mod conversation_store;
pub mod error;
pub mod llama_client;
pub mod openai_types;
pub mod system_prompt;
pub mod tool_registry;
pub mod translate_request;
pub mod translate_response;

use crate::conversation_store::ConversationStore;

/// The stream type produced by the local backend.
///
/// Items are individual `ResponseEvent`s; errors flow as `anyhow::Error` so
/// the upstream shim can wrap them in its own `Arc<AIApiError>` without our
/// crate having to depend on `app`.
pub type DispatchStream = BoxStream<'static, Result<ResponseEvent, anyhow::Error>>;

/// Process-wide conversation store. Each Warp process keeps one in-memory
/// store shared across all turns of all conversations; cloning is cheap
/// (Arc<DashMap> internally). On first call we attempt to hydrate it from
/// `$XDG_STATE_HOME/warp-llama/conversations/` (or `~/.local/state/...`);
/// every subsequent upsert is also persisted there.
fn store() -> ConversationStore {
    static STORE: OnceLock<ConversationStore> = OnceLock::new();
    STORE
        .get_or_init(|| {
            let dir = persist_dir();
            ConversationStore::load_from_disk(&dir)
        })
        .clone()
}

fn persist_dir() -> std::path::PathBuf {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| {
                let mut p = std::path::PathBuf::from(h);
                p.push(".local/state");
                p
            })
        })
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
    base.join("warp-llama").join("conversations")
}

/// Returns `Some(stream)` if `WARP_LLAMA_URL` is set, otherwise `None`.
///
/// Callers fall through to the real backend on `None`.
pub fn dispatch_if_enabled(request: &Request) -> Option<DispatchStream> {
    let cfg = config::Config::from_env()?;
    tracing::info!(
        target: "llama_backend",
        url = %cfg.url,
        model = %cfg.model,
        "dispatch_if_enabled: handling turn locally"
    );

    let store = store();
    let request = request.clone();

    let stream = async_stream::stream! {
        // 1. Translate inbound proto Request → OpenAI ChatCompletionRequest.
        let (openai_req, conversation_id) = match translate_request::proto_request_to_openai(
            &request,
            &store,
            system_prompt::WARP_SYSTEM_PROMPT,
            &cfg.model,
        ) {
            Ok(t) => t,
            Err(e) => {
                yield Ok(synthetic_finished_internal_error(&e.to_string()));
                return;
            }
        };

        // 2. Open SSE stream from llama-server.
        let client = llama_client::LlamaClient::new(cfg.url.clone(), cfg.api_key.clone());
        let raw_sse = match client.stream_completion(&openai_req) {
            Ok(s) => s,
            Err(e) => {
                yield Ok(synthetic_finished_internal_error(&e.to_string()));
                return;
            }
        };

        // 3. Translate OpenAI chunks → Warp `ResponseEvent`s. Pass the
        //    store so the translator can persist the assistant turn (text
        //    + tool_calls) for use in the next follow-up turn.
        let mut events = translate_response::openai_sse_to_proto_events(
            raw_sse,
            conversation_id.clone(),
            Some((store.clone(), conversation_id)),
        );
        while let Some(ev) = events.next().await {
            yield ev;
        }
    };
    Some(Box::pin(stream))
}

/// Build a `StreamFinished::InternalError` ResponseEvent for the case where
/// we can't even start the local-backend dance (request translation failed,
/// initial HTTP setup failed, etc.).
fn synthetic_finished_internal_error(msg: &str) -> ResponseEvent {
    use warp_multi_agent_api::response_event::stream_finished::{InternalError, Reason};
    use warp_multi_agent_api::response_event::{StreamFinished, Type as REType};
    ResponseEvent {
        r#type: Some(REType::Finished(StreamFinished {
            reason: Some(Reason::InternalError(InternalError {
                message: format!("llama_backend: {msg}"),
            })),
            ..Default::default()
        })),
    }
}
