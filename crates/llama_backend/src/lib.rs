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

#![deny(unused_must_use)]

use futures::stream::BoxStream;
use futures::StreamExt;
use warp_multi_agent_api::{Request, ResponseEvent};

pub mod config;

/// The stream type produced by the local backend.
///
/// Items are individual `ResponseEvent`s; errors flow as `anyhow::Error` so
/// the upstream shim can wrap them in its own `Arc<AIApiError>` without our
/// crate having to depend on `app`.
pub type DispatchStream = BoxStream<'static, Result<ResponseEvent, anyhow::Error>>;

/// Returns `Some(stream)` if `WARP_LLAMA_URL` is set, otherwise `None`.
///
/// Callers fall through to the real backend on `None`. The `_request` will be
/// consumed in subsequent phases; for now Phase 1 returns an empty stream as
/// the dispatch-routing test fixture.
pub fn dispatch_if_enabled(_request: &Request) -> Option<DispatchStream> {
    let _cfg = config::Config::from_env()?;
    tracing::info!(
        target: "llama_backend",
        "dispatch_if_enabled: WARP_LLAMA_URL set; returning local-backend stream"
    );
    Some(futures::stream::empty().boxed())
}
