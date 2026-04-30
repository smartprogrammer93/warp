//! In-memory conversation history, keyed by Warp's `conversation_id`.
//!
//! Warp's real backend keeps history server-side and the client only sends
//! the new turn's input + the existing `conversation_id`. Our local backend
//! has to reconstruct that ourselves: every turn we look up the prior
//! messages, append the new ones, and send the full transcript to
//! llama-server.
//!
//! Persistence to disk lands in Phase 10.3.

use crate::openai_types::{OpenAIMessage, OpenAIRole};
use dashmap::DashMap;
use std::sync::Arc;

/// One conversation's full message history (system prompt + all turns).
#[derive(Clone, Debug)]
pub struct Conversation {
    pub conversation_id: String,
    pub messages: Vec<OpenAIMessage>,
}

impl Conversation {
    pub fn new(conversation_id: String, system_prompt: String) -> Self {
        Self {
            conversation_id,
            messages: vec![OpenAIMessage {
                role: OpenAIRole::System,
                content: Some(system_prompt),
                tool_calls: None,
                tool_call_id: None,
            }],
        }
    }

    pub fn append_user(&mut self, text: String) {
        self.messages.push(OpenAIMessage {
            role: OpenAIRole::User,
            content: Some(text),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    pub fn append_tool_result(&mut self, tool_call_id: String, content: String) {
        self.messages.push(OpenAIMessage {
            role: OpenAIRole::Tool,
            content: Some(content),
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
        });
    }

    pub fn append_assistant(&mut self, msg: OpenAIMessage) {
        debug_assert!(matches!(msg.role, OpenAIRole::Assistant));
        self.messages.push(msg);
    }
}

/// Thread-safe per-process conversation store.
///
/// Keyed on `conversation_id`. The store is a process-global singleton
/// (initialized once in `lib.rs`); cloning it cheaply shares the same
/// underlying `DashMap`.
#[derive(Clone, Default)]
pub struct ConversationStore {
    inner: Arc<DashMap<String, Conversation>>,
}

impl ConversationStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fetch the conversation, or create one (with the system prompt as the
    /// first message) if `conversation_id` is unknown. Returns a clone so the
    /// caller can mutate freely; the caller is expected to call `upsert` to
    /// commit changes back.
    pub fn get_or_init(&self, conversation_id: &str, system_prompt: &str) -> Conversation {
        self.inner
            .entry(conversation_id.to_string())
            .or_insert_with(|| {
                Conversation::new(conversation_id.to_string(), system_prompt.to_string())
            })
            .clone()
    }

    pub fn upsert(&self, conv: Conversation) {
        self.inner.insert(conv.conversation_id.clone(), conv);
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_get_initializes_with_system_prompt() {
        let store = ConversationStore::new();
        let conv = store.get_or_init("c-1", "you are warp");
        assert_eq!(conv.conversation_id, "c-1");
        assert_eq!(conv.messages.len(), 1);
        assert_eq!(conv.messages[0].role, OpenAIRole::System);
        assert_eq!(conv.messages[0].content.as_deref(), Some("you are warp"));
    }

    #[test]
    fn append_user_message_grows_history() {
        let store = ConversationStore::new();
        let mut conv = store.get_or_init("c-1", "sys");
        conv.append_user("hello".into());
        store.upsert(conv);
        let again = store.get_or_init("c-1", "ignored on second get");
        assert_eq!(again.messages.len(), 2);
        assert_eq!(again.messages[1].role, OpenAIRole::User);
        assert_eq!(again.messages[1].content.as_deref(), Some("hello"));
    }

    #[test]
    fn second_get_does_not_reset_history() {
        let store = ConversationStore::new();
        let mut conv = store.get_or_init("c-1", "sys-A");
        conv.append_user("first".into());
        store.upsert(conv);
        // A different system prompt on second get should be IGNORED — the
        // stored conversation already exists.
        let again = store.get_or_init("c-1", "sys-B");
        assert_eq!(again.messages[0].content.as_deref(), Some("sys-A"));
    }

    #[test]
    fn distinct_conversations_dont_alias() {
        let store = ConversationStore::new();
        let mut a = store.get_or_init("c-A", "sys");
        a.append_user("hello A".into());
        store.upsert(a);

        let mut b = store.get_or_init("c-B", "sys");
        b.append_user("hello B".into());
        store.upsert(b);

        assert_eq!(store.len(), 2);
        let a = store.get_or_init("c-A", "sys");
        assert_eq!(a.messages[1].content.as_deref(), Some("hello A"));
        let b = store.get_or_init("c-B", "sys");
        assert_eq!(b.messages[1].content.as_deref(), Some("hello B"));
    }

    #[test]
    fn append_tool_result_carries_call_id() {
        let store = ConversationStore::new();
        let mut conv = store.get_or_init("c-1", "sys");
        conv.append_tool_result("call-7".into(), r#"{"exit_code":0}"#.into());
        store.upsert(conv);
        let again = store.get_or_init("c-1", "sys");
        let last = again.messages.last().unwrap();
        assert_eq!(last.role, OpenAIRole::Tool);
        assert_eq!(last.tool_call_id.as_deref(), Some("call-7"));
        assert_eq!(last.content.as_deref(), Some(r#"{"exit_code":0}"#));
    }

    #[test]
    fn store_is_clone_share() {
        // Cloning the store must share state.
        let a = ConversationStore::new();
        let b = a.clone();
        let mut conv = a.get_or_init("c-1", "sys");
        conv.append_user("hi".into());
        a.upsert(conv);
        let from_b = b.get_or_init("c-1", "sys");
        assert_eq!(from_b.messages.len(), 2);
    }
}
