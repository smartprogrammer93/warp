use thiserror::Error;

/// Errors produced by the local-backend dispatch path.
///
/// The `dispatch_if_enabled` stream uses `anyhow::Error` for crate-boundary
/// simplicity; this enum is the *internal* taxonomy that gets wrapped on the
/// way out. It's also what Phase 10.2 will pattern-match on to map specific
/// failure modes to specific `StreamFinished` reason variants.
#[derive(Debug, Error)]
pub enum LlamaBackendError {
    /// Couldn't reach llama-server (connection refused / DNS / TLS / etc.).
    #[error("transport: {0}")]
    Transport(String),

    /// llama-server responded with a non-2xx status.
    #[error("HTTP {status}: {body}")]
    HttpStatus { status: u16, body: String },

    /// Auth rejected. Maps to StreamFinished::InvalidApiKey in Phase 10.2.
    #[error("auth rejected: {0}")]
    InvalidApiKey(String),

    /// SSE-stream-level error (mid-stream disconnect, framing problem).
    #[error("sse: {0}")]
    Sse(String),

    /// We received an SSE chunk we couldn't parse as a ChatCompletionChunk.
    #[error("bad chunk: {0}")]
    BadPayload(String),

    /// Stream ended without a finish_reason. Should never happen with a
    /// well-behaved server.
    #[error("stream ended unexpectedly without a finish_reason")]
    EarlyEof,

    /// Catch-all for translation/encoding errors inside the crate.
    #[error("internal: {0}")]
    Internal(String),
}

impl From<reqwest::Error> for LlamaBackendError {
    fn from(e: reqwest::Error) -> Self {
        LlamaBackendError::Transport(e.to_string())
    }
}
