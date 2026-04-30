//! Translate the OpenAI-format SSE chunk stream from llama-server into the
//! transactional `ResponseEvent` sequence Warp's UI consumes.
//!
//! Per the proto-types catalogue, the stream we synthesize per turn is:
//!
//! ```text
//! StreamInit { conversation_id, request_id, run_id }
//! ClientActions { BeginTransaction }
//! ClientActions { CreateTask { id: <task_uuid>, description: "Agent Mode" } }
//! ClientActions { AddMessagesToTask { task_id, messages: [seed empty AgentOutput] } }
//! // For each text delta (after dropping reasoning_content + stripping
//! // leading "</think>" from first content delta):
//! ClientActions { AppendToMessageContent { task_id, message:{id:<msg>, agent_output:{text:<delta>}}, mask:["agent_output.text"] } }
//! // For each completed tool call (assembled from streamed deltas):
//! ClientActions { AddMessagesToTask { task_id, messages:[Message{id:<new>, tool_call:{tool_call_id, <variant>}}] } }
//! ClientActions { CommitTransaction }
//! StreamFinished { Done | ReachedMaxTokenLimit | InternalError }
//! ```
//!
//! The translator drops `delta.reasoning_content` entirely (Qwen3.6's
//! `<think>` block — Warp's UI has no surface for it in v0.1) and strips a
//! leading `</think>\s*` artifact from the first non-empty `delta.content`
//! chunk.

use crate::conversation_store::ConversationStore;
use crate::error::LlamaBackendError;
use crate::openai_types::{
    ChatCompletionChunk, OpenAIFunctionCall, OpenAIMessage, OpenAIRole, OpenAIToolCall,
};
use crate::tool_registry;
use async_stream::stream;
use futures::Stream;
use std::collections::BTreeMap;
use std::pin::Pin;
use uuid::Uuid;
use warp_multi_agent_api::client_action::Action as CAction;
use warp_multi_agent_api::message::Message as MMsg;
use warp_multi_agent_api::message::{AgentOutput, ToolCall as ProtoToolCall};
use warp_multi_agent_api::response_event::stream_finished::Reason;
use warp_multi_agent_api::response_event::{
    stream_finished, ClientActions, StreamFinished, StreamInit, Type as REType,
};
use warp_multi_agent_api::{client_action, ClientAction, Message, ResponseEvent, Task};

/// Convert an OpenAI SSE chunk stream into a stream of Warp `ResponseEvent`s.
///
/// Items are `Result<ResponseEvent, anyhow::Error>` so the caller can keep
/// the same error-channel shape used by `dispatch_if_enabled`. Errors that
/// arise mid-stream are *also* mapped to a terminal `StreamFinished` event
/// before the stream ends, so the UI gets a clean shutdown.
pub fn openai_sse_to_proto_events<S>(
    sse: S,
    conversation_id: String,
    store: Option<(ConversationStore, String)>,
) -> Pin<Box<dyn Stream<Item = Result<ResponseEvent, anyhow::Error>> + Send>>
where
    S: Stream<Item = Result<String, LlamaBackendError>> + Send + 'static,
{
    use futures::StreamExt;

    let task_id = Uuid::new_v4().to_string();
    let agent_msg_id = Uuid::new_v4().to_string();

    let s = stream! {
        let mut accumulated_content = String::new();
        // ---- 1. StreamInit ----
        yield Ok(make_stream_init(&conversation_id));

        // ---- 2. BeginTransaction ----
        yield Ok(wrap_action(CAction::BeginTransaction(
            client_action::BeginTransaction::default(),
        )));

        // ---- 3. CreateTask ----
        yield Ok(wrap_action(CAction::CreateTask(client_action::CreateTask {
            task: Some(Task {
                id: task_id.clone(),
                description: "Agent Mode".to_string(),
                ..Default::default()
            }),
        })));

        // ---- 4. Seed empty AgentOutput message ----
        yield Ok(wrap_action(CAction::AddMessagesToTask(
            client_action::AddMessagesToTask {
                task_id: task_id.clone(),
                messages: vec![Message {
                    id: agent_msg_id.clone(),
                    task_id: task_id.clone(),
                    message: Some(MMsg::AgentOutput(AgentOutput {
                        text: String::new(),
                    })),
                    ..Default::default()
                }],
            },
        )));

        // ---- 5. Stream chunks ----
        let mut sse = Box::pin(sse);
        let mut tool_calls: BTreeMap<usize, PartialToolCall> = BTreeMap::new();
        let mut think_stripper = ThinkStripper::default();
        let mut finish_reason_seen: Option<String> = None;
        let mut error: Option<LlamaBackendError> = None;

        while let Some(item) = sse.next().await {
            let payload = match item {
                Ok(s) => s,
                Err(e) => { error = Some(e); break; }
            };
            let chunk: ChatCompletionChunk = match serde_json::from_str(&payload) {
                Ok(c) => c,
                Err(e) => {
                    error = Some(LlamaBackendError::BadPayload(e.to_string()));
                    break;
                }
            };
            for choice in chunk.choices {
                // Drop reasoning_content (Qwen3.6 think block) entirely.
                if let Some(text_raw) = choice.delta.content {
                    let text = think_stripper.process(&text_raw);
                    if !text.is_empty() {
                        accumulated_content.push_str(&text);
                        yield Ok(wrap_action(CAction::AppendToMessageContent(
                            make_append_text(&task_id, &agent_msg_id, &text),
                        )));
                    }
                }
                if let Some(deltas) = choice.delta.tool_calls {
                    for d in deltas {
                        let entry = tool_calls
                            .entry(d.index)
                            .or_insert_with(PartialToolCall::default);
                        if let Some(id) = d.id { entry.id = id; }
                        if let Some(f) = d.function {
                            if let Some(name) = f.name { entry.name = name; }
                            if let Some(args) = f.arguments {
                                entry.args.push_str(&args);
                            }
                        }
                    }
                }
                if let Some(reason) = choice.finish_reason {
                    finish_reason_seen = Some(reason);
                }
            }
        }

        // ---- 5.5 Flush stripper buffer if model never closed </think>
        // (it didn't use thinking, so everything was real content). ----
        let flushed = think_stripper.flush();
        if !flushed.is_empty() {
            accumulated_content.push_str(&flushed);
            yield Ok(wrap_action(CAction::AppendToMessageContent(
                make_append_text(&task_id, &agent_msg_id, &flushed),
            )));
        }

        // ---- 6. Emit assembled tool calls (if any) ----
        for (_, partial) in tool_calls.iter() {
            match make_tool_call_message(&task_id, partial) {
                Ok(msg) => {
                    yield Ok(wrap_action(CAction::AddMessagesToTask(
                        client_action::AddMessagesToTask {
                            task_id: task_id.clone(),
                            messages: vec![msg],
                        },
                    )));
                }
                Err(e) => {
                    tracing::warn!(target: "llama_backend",
                        tool=%partial.name, "could not translate tool call: {e}");
                    error = Some(LlamaBackendError::Internal(format!(
                        "could not translate tool call {:?}: {e}", partial.name
                    )));
                    break;
                }
            }
        }

        // ---- 6.5 Persist the assistant turn into the conversation store
        // so the next turn (with a ToolCallResult input) sees the prior
        // assistant message + tool_calls in OpenAI's expected order. Only
        // persist on the success path; on error the partial state is
        // discarded.
        if error.is_none() {
            if let Some((store, conv_id)) = &store {
                let mut conv = store.get_or_init(conv_id, "");
                let assistant_tool_calls: Vec<OpenAIToolCall> = tool_calls
                    .values()
                    .map(|p| OpenAIToolCall {
                        id: if p.id.is_empty() {
                            Uuid::new_v4().to_string()
                        } else {
                            p.id.clone()
                        },
                        kind: "function".to_string(),
                        function: OpenAIFunctionCall {
                            name: p.name.clone(),
                            arguments: p.args.clone(),
                        },
                    })
                    .collect();
                let content_opt = if accumulated_content.is_empty() {
                    None
                } else {
                    Some(accumulated_content.clone())
                };
                let tool_calls_opt = if assistant_tool_calls.is_empty() {
                    None
                } else {
                    Some(assistant_tool_calls)
                };
                conv.append_assistant(OpenAIMessage {
                    role: OpenAIRole::Assistant,
                    content: content_opt,
                    tool_calls: tool_calls_opt,
                    tool_call_id: None,
                });
                store.upsert(conv);
            }
        }

        // ---- 7. CommitTransaction (always, even on error) ----
        yield Ok(wrap_action(CAction::CommitTransaction(
            client_action::CommitTransaction::default(),
        )));

        // ---- 8. StreamFinished ----
        let finished = match (error, finish_reason_seen.as_deref()) {
            (Some(e), _) => make_stream_finished_for_error(&e),
            (None, Some("length")) => make_stream_finished(Reason::MaxTokenLimit(
                stream_finished::ReachedMaxTokenLimit::default(),
            )),
            (None, _) => make_stream_finished(Reason::Done(stream_finished::Done::default())),
        };
        yield Ok(finished);
    };

    Box::pin(s)
}

#[derive(Default, Debug)]
struct PartialToolCall {
    id: String,
    name: String,
    args: String,
}

fn wrap_action(action: CAction) -> ResponseEvent {
    ResponseEvent {
        r#type: Some(REType::ClientActions(ClientActions {
            actions: vec![ClientAction {
                action: Some(action),
            }],
        })),
    }
}

fn make_stream_init(conversation_id: &str) -> ResponseEvent {
    ResponseEvent {
        r#type: Some(REType::Init(StreamInit {
            conversation_id: conversation_id.to_string(),
            request_id: Uuid::new_v4().to_string(),
            run_id: Uuid::new_v4().to_string(),
        })),
    }
}

fn make_append_text(
    task_id: &str,
    msg_id: &str,
    delta: &str,
) -> client_action::AppendToMessageContent {
    client_action::AppendToMessageContent {
        task_id: task_id.to_string(),
        message: Some(Message {
            id: msg_id.to_string(),
            task_id: task_id.to_string(),
            message: Some(MMsg::AgentOutput(AgentOutput {
                text: delta.to_string(),
            })),
            ..Default::default()
        }),
        mask: Some(prost_types::FieldMask {
            paths: vec!["agent_output.text".to_string()],
        }),
    }
}

fn make_tool_call_message(task_id: &str, partial: &PartialToolCall) -> anyhow::Result<Message> {
    let tool = tool_registry::build_proto_tool(&partial.name, &partial.args)?
        .ok_or_else(|| anyhow::anyhow!("unknown tool: {}", partial.name))?;
    let tool_call_id = if partial.id.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        partial.id.clone()
    };
    Ok(Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_string(),
        message: Some(MMsg::ToolCall(ProtoToolCall {
            tool_call_id,
            tool: Some(tool),
        })),
        ..Default::default()
    })
}

fn make_stream_finished(reason: Reason) -> ResponseEvent {
    ResponseEvent {
        r#type: Some(REType::Finished(StreamFinished {
            reason: Some(reason),
            ..Default::default()
        })),
    }
}

fn make_stream_finished_for_error(e: &LlamaBackendError) -> ResponseEvent {
    let reason = match e {
        LlamaBackendError::InvalidApiKey(body) => {
            Reason::InvalidApiKey(stream_finished::InvalidApiKey {
                provider: warp_multi_agent_api::LlmProvider::Unknown as i32,
                model_name: body.clone(),
            })
        }
        LlamaBackendError::Transport(_) => Reason::LlmUnavailable(stream_finished::LlmUnavailable::default()),
        other => Reason::InternalError(stream_finished::InternalError {
            message: other.to_string(),
        }),
    };
    make_stream_finished(reason)
}

/// Stateful filter that drops Qwen3.6's pre-`</think>` content noise.
///
/// Qwen3.6 (with default reasoning formatting) emits its think block partly
/// as `delta.reasoning_content` *and* partly as `delta.content` followed by
/// a literal `</think>` close tag. Stripping only the first delta isn't
/// enough — the close tag arrives in a later content chunk. So:
///
/// - We buffer all content until we see `</think>`. Suffix (after the tag,
///   left-trimmed) is what gets emitted as the model's actual answer.
/// - If the stream finishes without ever seeing `</think>`, we flush the
///   buffer as-is — the model didn't use thinking and everything is real
///   content.
#[derive(Default)]
struct ThinkStripper {
    close_think_seen: bool,
    buffer: String,
    emitted_any: bool,
}

impl ThinkStripper {
    fn process(&mut self, delta: &str) -> String {
        if self.close_think_seen {
            // After the close tag, lstrip until we've emitted real content.
            if self.emitted_any {
                return delta.to_string();
            }
            let lstripped = delta.trim_start();
            if lstripped.is_empty() {
                return String::new();
            }
            self.emitted_any = true;
            return lstripped.to_string();
        }
        self.buffer.push_str(delta);
        if let Some(pos) = self.buffer.find("</think>") {
            let suffix = self.buffer.split_off(pos + "</think>".len());
            self.buffer.clear();
            self.close_think_seen = true;
            let lstripped = suffix.trim_start();
            if lstripped.is_empty() {
                return String::new();
            }
            self.emitted_any = true;
            return lstripped.to_string();
        }
        String::new()
    }

    /// Called at end of stream. If we never saw `</think>`, the model didn't
    /// use thinking and everything we buffered is real content.
    fn flush(&mut self) -> String {
        if self.close_think_seen {
            return String::new();
        }
        self.close_think_seen = true;
        let buf = std::mem::take(&mut self.buffer);
        if !buf.is_empty() {
            self.emitted_any = true;
        }
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn sse(jsons: &[&str]) -> impl Stream<Item = Result<String, LlamaBackendError>> {
        let v: Vec<_> = jsons
            .iter()
            .map(|s| Ok::<_, LlamaBackendError>(s.to_string()))
            .collect();
        futures::stream::iter(v)
    }

    fn collect_all(
        s: impl Stream<Item = Result<ResponseEvent, anyhow::Error>>,
    ) -> Vec<ResponseEvent> {
        futures::executor::block_on(async {
            futures::pin_mut!(s);
            let mut out = vec![];
            while let Some(item) = s.next().await {
                out.push(item.expect("stream item"));
            }
            out
        })
    }

    #[test]
    fn think_stripper_buffers_until_close_then_emits_suffix() {
        let mut s = ThinkStripper::default();
        // Pre-</think> content: buffered, nothing emitted.
        assert_eq!(s.process("PONG\n"), "");
        // Close tag arrives mid-chunk: suffix after </think> emitted, lstripped.
        assert_eq!(s.process("</think>\n\nFINAL"), "FINAL");
        // After close: deltas pass through unchanged.
        assert_eq!(s.process(" answer"), " answer");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn think_stripper_flushes_buffer_when_close_never_arrives() {
        let mut s = ThinkStripper::default();
        // Model never uses </think> — everything we buffered should flush as content.
        assert_eq!(s.process("hello"), "");
        assert_eq!(s.process(" world"), "");
        assert_eq!(s.flush(), "hello world");
    }

    #[test]
    fn think_stripper_handles_close_tag_split_across_chunks() {
        let mut s = ThinkStripper::default();
        assert_eq!(s.process("...thinking</thi"), "");
        assert_eq!(s.process("nk>"), "");
        assert_eq!(s.process("\n\nactual"), "actual");
    }

    #[test]
    fn think_stripper_passes_close_after_already_passed() {
        let mut s = ThinkStripper::default();
        s.process("</think>real");
        // Subsequent chunks should pass even if they contain </think> literally.
        assert_eq!(s.process(" more"), " more");
    }

    #[test]
    fn first_event_is_stream_init() {
        let stream = openai_sse_to_proto_events(sse(&[]), "conv-1".into(), None);
        let events = collect_all(stream);
        assert!(events.len() >= 4); // init + begin + create + seed + commit + finished
        let first = &events[0];
        match &first.r#type {
            Some(REType::Init(init)) => assert_eq!(init.conversation_id, "conv-1"),
            other => panic!("expected init, got {other:?}"),
        }
    }

    #[test]
    fn empty_stream_emits_full_transactional_envelope_and_done() {
        let stream = openai_sse_to_proto_events(sse(&[]), "conv-1".into(), None);
        let events = collect_all(stream);
        // init + begin + create + seed + commit + finished = 6
        assert_eq!(events.len(), 6, "got {events:?}");
        assert!(matches!(events[0].r#type, Some(REType::Init(_))));
        match &events[5].r#type {
            Some(REType::Finished(f)) => {
                assert!(matches!(f.reason, Some(Reason::Done(_))));
            }
            other => panic!("expected Finished::Done, got {other:?}"),
        }
    }

    #[test]
    fn reasoning_content_deltas_are_dropped() {
        let stream = openai_sse_to_proto_events(
            sse(&[
                r#"{"choices":[{"delta":{"reasoning_content":"thinking..."}}]}"#,
                r#"{"choices":[{"finish_reason":"stop","delta":{}}]}"#,
            ]),
            "conv-1".into(),
            None,
        );
        let events = collect_all(stream);
        // Should see no AppendToMessageContent at all.
        let appends = events
            .iter()
            .filter(|e| matches!(
                &e.r#type,
                Some(REType::ClientActions(ca)) if ca.actions.iter().any(|a| matches!(a.action, Some(CAction::AppendToMessageContent(_))))
            ))
            .count();
        assert_eq!(appends, 0);
    }

    /// Helper: extract concatenated AgentOutput text from a stream of events.
    fn collect_text(events: &[ResponseEvent]) -> String {
        let mut out = String::new();
        for e in events {
            let Some(REType::ClientActions(ca)) = &e.r#type else { continue };
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

    #[test]
    fn pre_close_think_content_is_dropped_along_with_the_tag() {
        // Mirrors what Qwen3.6 actually emits: thinking tokens dump into
        // delta.content, followed by </think>, followed by the real answer.
        let stream = openai_sse_to_proto_events(
            sse(&[
                r#"{"choices":[{"delta":{"content":"PONG\n"}}]}"#,
                r#"{"choices":[{"delta":{"content":"</think>\n\n"}}]}"#,
                r#"{"choices":[{"delta":{"content":"PONG"}}]}"#,
                r#"{"choices":[{"finish_reason":"stop","delta":{}}]}"#,
            ]),
            "conv-1".into(),
            None,
        );
        let events = collect_all(stream);
        // The first "PONG\n" is in the think-section and should be dropped.
        // The "</think>" tag itself is dropped. Only the final "PONG" is emitted.
        assert_eq!(collect_text(&events), "PONG");
    }

    #[test]
    fn no_think_tag_emits_all_content() {
        // If the model never emits </think>, nothing in delta.content should
        // be dropped — flush at end-of-stream returns the buffer.
        let stream = openai_sse_to_proto_events(
            sse(&[
                r#"{"choices":[{"delta":{"content":"hello"}}]}"#,
                r#"{"choices":[{"delta":{"content":" world"}}]}"#,
                r#"{"choices":[{"finish_reason":"stop","delta":{}}]}"#,
            ]),
            "conv-1".into(),
            None,
        );
        let events = collect_all(stream);
        assert_eq!(collect_text(&events), "hello world");
    }

    #[test]
    fn text_deltas_concat_via_appends() {
        let stream = openai_sse_to_proto_events(
            sse(&[
                r#"{"choices":[{"delta":{"content":"hello"}}]}"#,
                r#"{"choices":[{"delta":{"content":" world"}}]}"#,
                r#"{"choices":[{"finish_reason":"stop","delta":{}}]}"#,
            ]),
            "conv-1".into(),
            None,
        );
        let events = collect_all(stream);
        let combined: String = events
            .iter()
            .filter_map(|e| match &e.r#type {
                Some(REType::ClientActions(ca)) => Some(ca),
                _ => None,
            })
            .flat_map(|ca| ca.actions.iter())
            .filter_map(|a| match &a.action {
                Some(CAction::AppendToMessageContent(ap)) => ap
                    .message
                    .as_ref()
                    .and_then(|m| m.message.as_ref())
                    .and_then(|mm| match mm {
                        MMsg::AgentOutput(ao) => Some(ao.text.clone()),
                        _ => None,
                    }),
                _ => None,
            })
            .collect();
        assert_eq!(combined, "hello world");
    }

    #[test]
    fn streamed_tool_call_is_assembled_and_emitted_as_run_shell_command() {
        let stream = openai_sse_to_proto_events(
            sse(&[
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"run_shell_command","arguments":"{"}}]}}]}"#,
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"command\":\"ls /tmp\"}"}}]}}]}"#,
                r#"{"choices":[{"finish_reason":"tool_calls","delta":{}}]}"#,
            ]),
            "conv-1".into(),
            None,
        );
        let events = collect_all(stream);
        // Find the AddMessagesToTask containing the tool call (NOT the seed AgentOutput one).
        let tool_calls: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.r#type {
                Some(REType::ClientActions(ca)) => Some(ca),
                _ => None,
            })
            .flat_map(|ca| ca.actions.iter())
            .filter_map(|a| match &a.action {
                Some(CAction::AddMessagesToTask(am)) => Some(am),
                _ => None,
            })
            .flat_map(|am| am.messages.iter())
            .filter_map(|m| match &m.message {
                Some(MMsg::ToolCall(tc)) => Some(tc),
                _ => None,
            })
            .collect();
        assert_eq!(tool_calls.len(), 1, "got {tool_calls:?}");
        assert_eq!(tool_calls[0].tool_call_id, "call-1");
        match tool_calls[0].tool.as_ref() {
            Some(warp_multi_agent_api::message::tool_call::Tool::RunShellCommand(rsc)) => {
                assert_eq!(rsc.command, "ls /tmp");
            }
            other => panic!("expected RunShellCommand, got {other:?}"),
        }
    }

    #[test]
    fn assistant_message_persisted_to_store_with_tool_calls() {
        use crate::conversation_store::ConversationStore;
        let store = ConversationStore::new();
        // Pre-seed a conversation so get_or_init has something to extend.
        let _ = store.get_or_init("conv-tc", "sys");

        let stream = openai_sse_to_proto_events(
            sse(&[
                r#"{"choices":[{"delta":{"content":"Sure, listing now."}}]}"#,
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-7","type":"function","function":{"name":"run_shell_command","arguments":"{\"command\":\"ls\"}"}}]}}]}"#,
                r#"{"choices":[{"finish_reason":"tool_calls","delta":{}}]}"#,
            ]),
            "conv-tc".into(),
            Some((store.clone(), "conv-tc".into())),
        );
        let _ = collect_all(stream);

        let conv = store.get_or_init("conv-tc", "sys");
        let last = conv.messages.last().unwrap();
        assert_eq!(last.role, OpenAIRole::Assistant);
        assert_eq!(last.content.as_deref(), Some("Sure, listing now."));
        let tcs = last.tool_calls.as_ref().expect("tool_calls populated");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "call-7");
        assert_eq!(tcs[0].function.name, "run_shell_command");
        assert_eq!(tcs[0].function.arguments, r#"{"command":"ls"}"#);
    }

    #[test]
    fn mid_stream_invalid_api_key_error_maps_to_invalid_api_key_reason() {
        let stream = openai_sse_to_proto_events(
            futures::stream::iter(vec![Err::<String, _>(
                LlamaBackendError::InvalidApiKey("nope".into()),
            )]),
            "conv-1".into(),
            None,
        );
        let events = collect_all(stream);
        let last = events.last().unwrap();
        match &last.r#type {
            Some(REType::Finished(f)) => {
                assert!(matches!(f.reason, Some(Reason::InvalidApiKey(_))));
            }
            other => panic!("expected Finished::InvalidApiKey, got {other:?}"),
        }
    }

    #[test]
    fn mid_stream_transport_error_maps_to_llm_unavailable() {
        let stream = openai_sse_to_proto_events(
            futures::stream::iter(vec![Err::<String, _>(
                LlamaBackendError::Transport("connection refused".into()),
            )]),
            "conv-1".into(),
            None,
        );
        let events = collect_all(stream);
        let last = events.last().unwrap();
        match &last.r#type {
            Some(REType::Finished(f)) => {
                assert!(matches!(f.reason, Some(Reason::LlmUnavailable(_))));
            }
            other => panic!("expected LlmUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn finish_reason_length_maps_to_max_token_limit() {
        let stream = openai_sse_to_proto_events(
            sse(&[r#"{"choices":[{"finish_reason":"length","delta":{}}]}"#]),
            "conv-1".into(),
            None,
        );
        let events = collect_all(stream);
        let last = events.last().unwrap();
        match &last.r#type {
            Some(REType::Finished(f)) => {
                assert!(matches!(f.reason, Some(Reason::MaxTokenLimit(_))));
            }
            other => panic!("expected Finished::MaxTokenLimit, got {other:?}"),
        }
    }
}
