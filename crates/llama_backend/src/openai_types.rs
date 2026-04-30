//! Minimal OpenAI Chat-Completions wire types.
//!
//! We don't pull in `async-openai` because it's heavy and brings async-runtime
//! coupling we don't need; only the subset of fields actually used over the
//! wire by llama-server is modeled here.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIMessage {
    pub role: OpenAIRole,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type", default = "default_tool_call_kind")]
    pub kind: String,
    pub function: OpenAIFunctionCall,
}

fn default_tool_call_kind() -> String {
    "function".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    /// Raw JSON-encoded args, per the OpenAI spec.
    pub arguments: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpenAIToolDef {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OpenAIFunctionDef,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpenAIFunctionDef {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the function's parameters.
    pub parameters: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OpenAIToolDef>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

// ---------------- streaming chunk types ----------------

/// One SSE chunk in `data: {...}` form from `/v1/chat/completions?stream=true`.
#[derive(Clone, Debug, Deserialize)]
pub struct ChatCompletionChunk {
    pub choices: Vec<ChunkChoice>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChunkChoice {
    pub delta: ChunkDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Default)]
pub struct ChunkDelta {
    #[serde(default)]
    pub content: Option<String>,
    /// Qwen3.6 emits its `<think>` block as `reasoning_content` deltas.
    /// The translator drops these (Warp's UI has no thinking-block surface in v0.1).
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChunkToolCallDelta>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChunkToolCallDelta {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<ChunkFunctionDelta>,
}

#[derive(Clone, Debug, Deserialize, Default)]
pub struct ChunkFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_serialization_skips_none_fields() {
        let msg = OpenAIMessage {
            role: OpenAIRole::User,
            content: Some("hi".into()),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"role":"user","content":"hi"}"#);
    }

    #[test]
    fn tool_message_carries_tool_call_id() {
        let msg = OpenAIMessage {
            role: OpenAIRole::Tool,
            content: Some("ok".into()),
            tool_calls: None,
            tool_call_id: Some("call-abc".into()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""role":"tool""#));
        assert!(json.contains(r#""tool_call_id":"call-abc""#));
        let back: OpenAIMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool_call_id.as_deref(), Some("call-abc"));
        assert_eq!(back.role, OpenAIRole::Tool);
    }

    #[test]
    fn chunk_decodes_text_delta() {
        let raw = r#"{"choices":[{"delta":{"role":"assistant","content":"hello"}}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
        assert!(chunk.choices[0].delta.reasoning_content.is_none());
        assert!(chunk.choices[0].delta.tool_calls.is_none());
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn chunk_decodes_reasoning_delta() {
        let raw = r#"{"choices":[{"delta":{"reasoning_content":"think"}}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(
            chunk.choices[0].delta.reasoning_content.as_deref(),
            Some("think")
        );
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn chunk_decodes_streamed_tool_call_first_delta() {
        let raw = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"X","type":"function","function":{"name":"run_shell_command","arguments":"{"}}]}}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let tcs = chunk.choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].index, 0);
        assert_eq!(tcs[0].id.as_deref(), Some("X"));
        let f = tcs[0].function.as_ref().unwrap();
        assert_eq!(f.name.as_deref(), Some("run_shell_command"));
        assert_eq!(f.arguments.as_deref(), Some("{"));
    }

    #[test]
    fn chunk_decodes_streamed_tool_call_subsequent_delta() {
        let raw =
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"cmd\":\"ls\"}"}}]}}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let tcs = chunk.choices[0].delta.tool_calls.as_ref().unwrap();
        assert!(tcs[0].id.is_none());
        let f = tcs[0].function.as_ref().unwrap();
        assert!(f.name.is_none());
        assert_eq!(f.arguments.as_deref(), Some(r#""cmd":"ls"}"#));
    }

    #[test]
    fn chunk_decodes_finish_reason_tool_calls() {
        let raw = r#"{"choices":[{"finish_reason":"tool_calls","delta":{}}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("tool_calls"));
    }
}
