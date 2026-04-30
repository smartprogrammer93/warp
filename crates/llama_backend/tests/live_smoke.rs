//! End-to-end smoke test against a real llama-server.
//!
//! Marked `#[ignore]` because it requires:
//! - Network access to CT501 (192.168.8.68:8080)
//! - The env vars `WARP_LLAMA_URL`, `WARP_LLAMA_MODEL`, `WARP_LLAMA_API_KEY` set
//! - Up-to-date crypto provider (`rustls::aws_lc_rs::default_provider()`)
//!
//! Run with:
//!
//! ```sh
//! WARP_LLAMA_URL=http://192.168.8.68:8080 \
//! WARP_LLAMA_MODEL=Qwen3.6-27B-UD-Q4_K_XL.gguf \
//! WARP_LLAMA_API_KEY=ncvbuI7MluQB8oC772W-tjsGdVlIKS0x39snaLqQ_eY \
//! cargo test -p llama_backend --test live_smoke -- --ignored --nocapture --test-threads=1
//! ```

use futures::StreamExt;
use llama_backend::dispatch_if_enabled;
use warp_multi_agent_api::client_action::Action as CAction;
use warp_multi_agent_api::message::Message as MMsg;
use warp_multi_agent_api::request::input::user_inputs::user_input::Input as UserInputKind;
use warp_multi_agent_api::request::input::user_inputs::UserInput;
use warp_multi_agent_api::request::input::{Type as InputType, UserInputs, UserQuery};
use warp_multi_agent_api::request::{Input, Metadata};
use warp_multi_agent_api::response_event::stream_finished::Reason;
use warp_multi_agent_api::response_event::Type as REType;
use warp_multi_agent_api::Request;

fn ensure_crypto_provider() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    });
}

fn make_user_query_request(text: &str) -> Request {
    Request {
        metadata: Some(Metadata {
            conversation_id: String::new(),
            ..Default::default()
        }),
        input: Some(Input {
            r#type: Some(InputType::UserInputs(UserInputs {
                inputs: vec![UserInput {
                    input: Some(UserInputKind::UserQuery(UserQuery {
                        query: text.to_string(),
                        ..Default::default()
                    })),
                }],
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires live llama-server; opt in via --ignored"]
async fn say_pong_against_real_llama_server() {
    ensure_crypto_provider();

    assert!(
        std::env::var("WARP_LLAMA_URL").is_ok(),
        "WARP_LLAMA_URL must be set"
    );

    let req = make_user_query_request(
        "Reply with the single word PONG and nothing else. No tools, no explanation.",
    );
    let stream = dispatch_if_enabled(&req).expect("WARP_LLAMA_URL should engage the backend");
    let events: Vec<_> = stream.collect().await;
    let events: Vec<_> = events
        .into_iter()
        .map(|e| e.expect("no stream-level errors"))
        .collect();

    eprintln!("=== {} events ===", events.len());
    for (i, e) in events.iter().enumerate() {
        eprintln!("[{i}] {:?}", e.r#type.as_ref().map(short_type_name));
    }

    // First event must be StreamInit.
    assert!(matches!(events[0].r#type, Some(REType::Init(_))), "first event not StreamInit");
    // Last event must be StreamFinished::Done (or MaxTokenLimit).
    let last = events.last().unwrap();
    match &last.r#type {
        Some(REType::Finished(f)) => match &f.reason {
            Some(Reason::Done(_)) | Some(Reason::MaxTokenLimit(_)) => {}
            other => panic!("unexpected finish reason: {other:?}"),
        },
        other => panic!("last event not Finished, got {other:?}"),
    }

    // Concatenate all AppendToMessageContent text deltas — should contain
    // "PONG" (case-insensitive, since the model may quote it).
    let combined = collect_appended_text(&events);
    eprintln!("=== model output ===\n{combined}\n=== end ===");
    assert!(
        combined.to_uppercase().contains("PONG"),
        "expected PONG in output, got {combined:?}"
    );
}

fn short_type_name(t: &REType) -> &'static str {
    match t {
        REType::Init(_) => "Init",
        REType::ClientActions(ca) => match ca.actions.first().and_then(|a| a.action.as_ref()) {
            Some(CAction::BeginTransaction(_)) => "Begin",
            Some(CAction::CreateTask(_)) => "CreateTask",
            Some(CAction::AddMessagesToTask(_)) => "AddMessages",
            Some(CAction::AppendToMessageContent(_)) => "AppendText",
            Some(CAction::CommitTransaction(_)) => "Commit",
            _ => "OtherAction",
        },
        REType::Finished(_) => "Finished",
    }
}

fn collect_appended_text(events: &[warp_multi_agent_api::ResponseEvent]) -> String {
    let mut out = String::new();
    for e in events {
        let Some(REType::ClientActions(ca)) = &e.r#type else {
            continue;
        };
        for a in &ca.actions {
            if let Some(CAction::AppendToMessageContent(ap)) = &a.action {
                if let Some(MMsg::AgentOutput(ao)) =
                    ap.message.as_ref().and_then(|m| m.message.as_ref())
                {
                    out.push_str(&ao.text);
                }
            }
        }
    }
    out
}
