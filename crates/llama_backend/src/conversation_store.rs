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
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One conversation's full message history (system prompt + all turns).
#[derive(Clone, Debug, Serialize, Deserialize)]
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
///
/// Optionally persists each conversation to a directory: each upsert
/// rewrites `<dir>/<conversation_id>.json` atomically (write to tempfile +
/// rename). On Warp restart, a fresh store can be initialized from the
/// directory via [`ConversationStore::load_from_disk`].
#[derive(Clone, Default)]
pub struct ConversationStore {
    inner: Arc<DashMap<String, Conversation>>,
    persist_dir: Option<PathBuf>,
}

impl ConversationStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a store that loads any existing conversations from `dir` and
    /// then persists every upsert back to it. If `dir` doesn't exist it is
    /// created. Load errors on individual files are logged and skipped (a
    /// corrupted state file should not prevent Warp from starting).
    pub fn load_from_disk(dir: &Path) -> Self {
        if let Err(e) = std::fs::create_dir_all(dir) {
            tracing::warn!(
                target: "llama_backend",
                dir = %dir.display(),
                "could not create persist dir: {e}; running without persistence"
            );
            return Self::new();
        }
        let inner: Arc<DashMap<String, Conversation>> = Arc::new(DashMap::new());
        match std::fs::read_dir(dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) != Some("json") {
                        continue;
                    }
                    match std::fs::read(&path)
                        .map_err(|e| e.to_string())
                        .and_then(|b| {
                            serde_json::from_slice::<Conversation>(&b).map_err(|e| e.to_string())
                        }) {
                        Ok(conv) => {
                            inner.insert(conv.conversation_id.clone(), conv);
                        }
                        Err(e) => {
                            tracing::warn!(
                                target: "llama_backend",
                                path = %path.display(),
                                "could not load conversation: {e}"
                            );
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "llama_backend",
                    dir = %dir.display(),
                    "could not read persist dir: {e}"
                );
            }
        }
        let count = inner.len();
        if count > 0 {
            tracing::info!(target: "llama_backend", count, "loaded conversations from disk");
        }
        Self {
            inner,
            persist_dir: Some(dir.to_path_buf()),
        }
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
        if let Some(dir) = &self.persist_dir {
            if let Err(e) = write_conversation_atomic(dir, &conv) {
                tracing::warn!(
                    target: "llama_backend",
                    conversation_id = %conv.conversation_id,
                    "could not persist conversation: {e}"
                );
            }
        }
        self.inner.insert(conv.conversation_id.clone(), conv);
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

fn write_conversation_atomic(dir: &Path, conv: &Conversation) -> std::io::Result<()> {
    let final_path = dir.join(format!("{}.json", sanitize_filename(&conv.conversation_id)));
    let tmp_path = dir.join(format!(
        ".{}.json.tmp",
        sanitize_filename(&conv.conversation_id)
    ));
    let bytes = serde_json::to_vec_pretty(conv)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(&tmp_path, bytes)?;
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(())
}

/// Conversation IDs are UUIDs we mint, but be defensive: only allow
/// [a-zA-Z0-9._-] in filenames so a malicious server-supplied id can't
/// escape the persist dir or clobber a sibling file.
fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
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

    #[test]
    fn persist_and_reload_roundtrip() {
        // Use a tempdir per test to avoid cross-test interference.
        let tmp = tempdir_for_test();
        // Phase A: write
        {
            let store = ConversationStore::load_from_disk(&tmp);
            let mut conv = store.get_or_init("conv-persist", "you are warp");
            conv.append_user("first message".into());
            conv.append_assistant(crate::openai_types::OpenAIMessage {
                role: crate::openai_types::OpenAIRole::Assistant,
                content: Some("first reply".into()),
                tool_calls: None,
                tool_call_id: None,
            });
            store.upsert(conv);
        }
        // The file should exist on disk now.
        let file = tmp.join("conv-persist.json");
        assert!(file.exists(), "json file should have been written");

        // Phase B: reload into a fresh store
        let store2 = ConversationStore::load_from_disk(&tmp);
        let conv = store2.get_or_init("conv-persist", "ignored on reload");
        assert_eq!(conv.messages.len(), 3);
        assert_eq!(conv.messages[0].content.as_deref(), Some("you are warp"));
        assert_eq!(conv.messages[1].content.as_deref(), Some("first message"));
        assert_eq!(conv.messages[2].content.as_deref(), Some("first reply"));
    }

    #[test]
    fn corrupted_file_does_not_prevent_load() {
        let tmp = tempdir_for_test();
        std::fs::write(tmp.join("garbage.json"), b"not json").unwrap();
        // Should not panic; should produce an empty store.
        let store = ConversationStore::load_from_disk(&tmp);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn sanitize_filename_replaces_unsafe_chars() {
        assert_eq!(sanitize_filename("abc-123_def.json"), "abc-123_def.json");
        assert_eq!(sanitize_filename("../etc/passwd"), ".._etc_passwd");
        assert_eq!(sanitize_filename("a/b/c"), "a_b_c");
    }

    fn tempdir_for_test() -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "warp-llama-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
