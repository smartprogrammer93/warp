//! Translate `warp_multi_agent_api::Request` into an
//! [`ChatCompletionRequest`] suitable for llama-server.
//!
//! This is the outbound half of the translator pair (the inbound half is
//! `translate_response.rs`). On every turn we:
//!
//! 1. Resolve the conversation id (mint a fresh UUID if the client sends an
//!    empty one — first turn).
//! 2. Look up (or initialize, with the system prompt) the conversation in
//!    the store.
//! 3. Walk `Request.input.type` and append the new turn's contributions to
//!    the stored history.
//! 4. Build the OpenAI request from the resulting history plus per-request
//!    tools.
//!
//! Per the plan, v0.1 only handles `Input::Type::UserInputs` containing
//! `UserQuery` and `ToolCallResult` variants; other input types return
//! `Err(...)` so the caller can emit a clean `StreamFinished::InternalError`.

use crate::conversation_store::ConversationStore;
use crate::openai_types::ChatCompletionRequest;
use crate::tool_registry::tools_for_request;
use anyhow::{anyhow, Result};
use uuid::Uuid;
use warp_multi_agent_api::request::input::user_inputs::user_input::Input as UserInputKind;
use warp_multi_agent_api::request::input::Type as InputType;
use warp_multi_agent_api::Request as ProtoRequest;

/// Default temperature passed to llama-server for chat completions.
///
/// Matches the user's existing CT501 deployment per `agents-ct501.md`
/// (Qwopus model-card "thinking + coding" preset; sampling is also enforced
/// server-side via the systemd unit's `--temp`/`--top-k`/`--top-p` flags, so
/// this is the field we send to be explicit but the server may override).
const DEFAULT_TEMPERATURE: f32 = 0.6;

/// Default max output tokens. Reasonable for a single agent turn including
/// thinking + tool calls. Not cumulative across turns — Qwen3.6 with 262144
/// ctx has plenty of headroom on the input side.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Convert a Warp protobuf request into an OpenAI chat-completions request,
/// using the conversation store to fill in prior history.
///
/// Returns the request to send to llama-server plus the resolved
/// `conversation_id` (the same one the client already had, or the freshly
/// minted UUID for first turns).
pub fn proto_request_to_openai(
    proto: &ProtoRequest,
    store: &ConversationStore,
    system_prompt: &str,
    model: &str,
) -> Result<(ChatCompletionRequest, String)> {
    // 1. Resolve / mint conversation id.
    let conversation_id = proto
        .metadata
        .as_ref()
        .map(|m| m.conversation_id.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    // 2. Get-or-init the stored conversation. (System prompt only used on
    // first init; ignored on subsequent gets per ConversationStore semantics.)
    let mut conv = store.get_or_init(&conversation_id, system_prompt);

    // 3. Walk the input.
    let input = proto
        .input
        .as_ref()
        .ok_or_else(|| anyhow!("Request.input is missing"))?;
    let kind = input
        .r#type
        .as_ref()
        .ok_or_else(|| anyhow!("Request.input.type is missing"))?;

    match kind {
        InputType::UserInputs(user_inputs) => {
            for ui in &user_inputs.inputs {
                let inner = ui
                    .input
                    .as_ref()
                    .ok_or_else(|| anyhow!("UserInput.input is missing"))?;
                match inner {
                    UserInputKind::UserQuery(q) => {
                        conv.append_user(q.query.clone());
                    }
                    UserInputKind::ToolCallResult(r) => {
                        // Phase 4 placeholder: serialize a minimal JSON.
                        // Replaced with full per-variant serialization in
                        // Phase 9.
                        let content = serde_json::to_string(&serde_json::json!({
                            "tool_call_id": r.tool_call_id,
                            "_phase4_placeholder": "Phase 9 will produce full JSON shapes per ToolCallResult variant",
                        }))
                        .unwrap_or_else(|_| "{}".to_string());
                        conv.append_tool_result(r.tool_call_id.clone(), content);
                    }
                    other => {
                        return Err(anyhow!(
                            "UserInput.input variant {:?} not yet supported by llama_backend (v0.1)",
                            std::mem::discriminant(other)
                        ));
                    }
                }
            }
        }
        other => {
            return Err(anyhow!(
                "Request.input.type variant {:?} not yet supported by llama_backend (v0.1; only UserInputs is handled)",
                std::mem::discriminant(other)
            ));
        }
    }

    // 4. Persist updated history before building the OpenAI request.
    store.upsert(conv.clone());

    // 5. Build the request.
    let openai = ChatCompletionRequest {
        model: model.to_string(),
        messages: conv.messages,
        tools: tools_for_request(proto),
        stream: true,
        max_tokens: Some(DEFAULT_MAX_TOKENS),
        temperature: Some(DEFAULT_TEMPERATURE),
    };

    Ok((openai, conversation_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_types::OpenAIRole;
    use warp_multi_agent_api::{
        request::{
            self,
            input::{user_inputs::{user_input, UserInput}, UserInputs, UserQuery},
        },
        ToolType,
    };

    fn settings_with_supported(supported: Vec<ToolType>) -> request::Settings {
        request::Settings {
            supported_tools: supported.into_iter().map(|t| t as i32).collect(),
            ..Default::default()
        }
    }

    fn fresh_user_query_request(text: &str) -> ProtoRequest {
        ProtoRequest {
            metadata: Some(request::Metadata {
                conversation_id: String::new(),
                ..Default::default()
            }),
            input: Some(request::Input {
                r#type: Some(InputType::UserInputs(UserInputs {
                    inputs: vec![UserInput {
                        input: Some(user_input::Input::UserQuery(UserQuery {
                            query: text.to_string(),
                            ..Default::default()
                        })),
                    }],
                })),
                ..Default::default()
            }),
            settings: Some(settings_with_supported(vec![])),
            ..Default::default()
        }
    }

    fn followup_user_query_request(conv_id: &str, text: &str) -> ProtoRequest {
        let mut req = fresh_user_query_request(text);
        req.metadata = Some(request::Metadata {
            conversation_id: conv_id.to_string(),
            ..Default::default()
        });
        req
    }

    #[test]
    fn fresh_turn_initializes_with_system_prompt_and_user_message() {
        let store = ConversationStore::new();
        let req = fresh_user_query_request("hello");
        let (openai, conv_id) =
            proto_request_to_openai(&req, &store, "you are warp", "qwen-test").unwrap();

        assert_eq!(openai.model, "qwen-test");
        assert!(openai.stream);
        assert_eq!(openai.messages.len(), 2);
        assert_eq!(openai.messages[0].role, OpenAIRole::System);
        assert_eq!(openai.messages[0].content.as_deref(), Some("you are warp"));
        assert_eq!(openai.messages[1].role, OpenAIRole::User);
        assert_eq!(openai.messages[1].content.as_deref(), Some("hello"));
        assert!(!conv_id.is_empty());
        assert_ne!(conv_id, ""); // non-empty UUID minted

        // Should include all 6 static tools (empty supported_tools = no filter).
        assert_eq!(openai.tools.len(), 6);
    }

    #[test]
    fn followup_turn_reuses_conversation_history() {
        let store = ConversationStore::new();
        let req1 = fresh_user_query_request("first");
        let (_, conv_id) =
            proto_request_to_openai(&req1, &store, "sys", "qwen").unwrap();

        // Simulate the model having replied — append an assistant message
        // (Phase 7 will do this via the inbound translator; for the test we
        // mutate the store directly).
        {
            let mut conv = store.get_or_init(&conv_id, "ignored");
            conv.append_assistant(crate::openai_types::OpenAIMessage {
                role: OpenAIRole::Assistant,
                content: Some("first reply".into()),
                tool_calls: None,
                tool_call_id: None,
            });
            store.upsert(conv);
        }

        let req2 = followup_user_query_request(&conv_id, "second");
        let (openai, conv_id2) =
            proto_request_to_openai(&req2, &store, "sys", "qwen").unwrap();
        assert_eq!(conv_id2, conv_id);
        // sys, user1, asst1, user2 = 4 messages
        assert_eq!(openai.messages.len(), 4);
        assert_eq!(openai.messages[3].role, OpenAIRole::User);
        assert_eq!(openai.messages[3].content.as_deref(), Some("second"));
    }

    #[test]
    fn supported_tools_filters_through() {
        let store = ConversationStore::new();
        let mut req = fresh_user_query_request("hi");
        req.settings = Some(settings_with_supported(vec![ToolType::Grep]));
        let (openai, _) = proto_request_to_openai(&req, &store, "sys", "m").unwrap();
        assert_eq!(openai.tools.len(), 1);
        assert_eq!(openai.tools[0].function.name, "grep");
    }

    #[test]
    fn unsupported_input_type_errors() {
        let store = ConversationStore::new();
        let req = ProtoRequest {
            metadata: Some(request::Metadata::default()),
            input: Some(request::Input {
                r#type: Some(InputType::ResumeConversation(
                    request::input::ResumeConversation::default(),
                )),
                ..Default::default()
            }),
            ..Default::default()
        };
        let err = proto_request_to_openai(&req, &store, "sys", "m").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not yet supported"), "got {msg}");
    }

    #[test]
    fn missing_input_errors() {
        let store = ConversationStore::new();
        let req = ProtoRequest {
            metadata: Some(request::Metadata::default()),
            input: None,
            ..Default::default()
        };
        let err = proto_request_to_openai(&req, &store, "sys", "m").unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn tool_call_result_is_appended_as_tool_message() {
        let store = ConversationStore::new();
        let conv_id = "c-1";
        // Pre-seed a turn so the store has a conversation.
        {
            let _ = store.get_or_init(conv_id, "sys");
        }
        let req = ProtoRequest {
            metadata: Some(request::Metadata {
                conversation_id: conv_id.to_string(),
                ..Default::default()
            }),
            input: Some(request::Input {
                r#type: Some(InputType::UserInputs(UserInputs {
                    inputs: vec![UserInput {
                        input: Some(user_input::Input::ToolCallResult(
                            request::input::ToolCallResult {
                                tool_call_id: "call-7".to_string(),
                                result: None,
                            },
                        )),
                    }],
                })),
                ..Default::default()
            }),
            settings: Some(settings_with_supported(vec![])),
            ..Default::default()
        };
        let (openai, returned_id) =
            proto_request_to_openai(&req, &store, "sys", "m").unwrap();
        assert_eq!(returned_id, conv_id);
        let last = openai.messages.last().unwrap();
        assert_eq!(last.role, OpenAIRole::Tool);
        assert_eq!(last.tool_call_id.as_deref(), Some("call-7"));
        assert!(last.content.as_deref().unwrap().contains("call-7"));
    }
}
