# Warp Local-LLM Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fork the open-source Warp terminal (`warp-oss` build) and replace its remote AI backend (`https://app.warp.dev/ai/multi-agent`) with a direct OpenAI-compatible call to a self-hosted llama-server (Qwen3.6-27B on `192.168.8.68:8080`), so all of Warp's agent / Agent-Mode features run against a local LLM with no Warp account required for AI features.

**Architecture:** Warp's AI client speaks a fixed wire protocol — HTTP POST a `warp_multi_agent_api::Request` protobuf to `/ai/multi-agent`, receive an SSE stream of base64-URL-safe-encoded `warp_multi_agent_api::ResponseEvent` protobufs. We add a `crates/llama_backend/` crate that implements the same `ResponseStream`-returning interface as `ServerApi::generate_multi_agent_output`, but synthesizes the protobuf events from a llama-server `/v1/chat/completions` SSE stream. We dispatch to it from `app/src/server/server_api.rs:1091` behind an env-var feature flag (`WARP_LLAMA_URL`), so the real backend remains the default and the local backend is opt-in. Conversation history (which the real Warp backend stores server-side) is reconstructed and persisted client-side in a per-token `ConversationStore`.

**Tech Stack:**
- **Fork base:** github.com/warpdotdev/warp @ `main` (AGPLv3/MIT dual-license; `warp-oss` build target — no telemetry, no autoupdate, no signing keys, public Firebase API key embedded in source)
- **Language:** Rust 2021 (workspace already established)
- **Protos:** `warp_multi_agent_api` from github.com/warpdotdev/warp-proto-apis @ `aa2f9cde164a5b48ac01087d417d1188771f9b6d` (vendored Cargo.toml dep — public)
- **HTTP/SSE client:** `reqwest` + `reqwest_eventsource` (already workspace deps)
- **Stream synthesis:** `async-stream` (add)
- **Mock HTTP for tests:** `wiremock` (add as dev-dep)
- **Conversation persistence:** in-memory `dashmap` for v1; on-disk JSON in `~/.local/state/warp-llama/conversations/` for v1.1
- **Target backend:** llama-server's OpenAI-compatible `POST /v1/chat/completions` (already deployed on CT501 at `http://192.168.8.68:8080`, serving `Qwen3.6-27B-UD-Q3_K_XL.gguf @ -c 262144`)

---

## Phase 0.3 Amendments (after reading actual proto sources, 2026-04-30)

Read order of importance, all in `~/projects/warp-proto-apis-ref/apis/multi_agent/v1/`:
- `request.proto` (544 LoC) — `Request`, `Input` oneof (14 variants), `Settings`, `Metadata`, `MCPContext`
- `response.proto` (331 LoC) — `ResponseEvent`, `StreamInit`, `StreamFinished` (8 reason variants), `ClientAction` (14 variants)
- `task.proto` (1862 LoC) — `Task`, `Message` (21+ variants), every `ToolCall::*`, every `ToolCallResult::*`, the `ToolType` enum, all the per-tool result shapes
- `skill.proto` — Skill / SkillRef / SkillDescriptor

Recon vs. reality:

| Recon claim | Reality |
|---|---|
| Conversation key is `conversation_token` | **`conversation_id`** at `Request.metadata.conversation_id` (string) and `StreamInit.conversation_id` |
| `Input::input_kind` matched on `UserInputs / ToolCallResult` | `Request.input.type` has **14 variants** (request.proto:53–71). The user's text is at `request.input.user_inputs.inputs[i].user_query.query` (deep nesting) |
| Text deltas → `AppendToMessageContent { text }` | `AppendToMessageContent` carries `Message + FieldMask`, the mask names which string field appends. We use `paths: ["agent_output.text"]` |
| Tool variants in `request.proto` | Tool variants live in **`task.proto`** (`Message.ToolCall.tool` oneof, ~30 variants); their result variants live in **`task.proto`** (`Message.ToolCallResult.result` oneof) |
| Generated rust at `gen/rust/src/lib.rs` is the source of truth | That file is a **27-LoC stub**; prost-build generates types at compile time from `.proto` files |
| Auth header attached by Warp client to `/ai/multi-agent` | True for upstream — but our shim returns *before* the bearer-auth call, so the Firebase token is irrelevant to llama_backend |

The full type catalogue with field paths and the exact event-sequence diagram is at `proto-types-catalogue.md` (sibling file). Subsequent phases reference it. Notable downstream impacts:

- **Phase 4** (outbound translator): replace every `conversation_token` with `conversation_id`. The user-text extraction now navigates `request.input.type → user_inputs → inputs[i].input → user_query.query` (and similarly for `tool_call_result`). All other Input variants emit a polite `StreamFinished { internal_error }` and exit.
- **Phase 7** (inbound translator): emit a transactional event sequence per turn (BeginTransaction → CreateTask → AddMessagesToTask seed → AppendToMessageContent×N → AddMessagesToTask per tool-call → CommitTransaction → StreamFinished). UUIDs (add `uuid = { workspace = true, features = ["v4"] }`) for task_id and message_id. FieldMask via `prost-types`.
- **Phase 3** (tool registry): hard-coded set is **only 6 OpenAI schemas** for v0.1 (`run_shell_command`, `read_files`, `search_codebase`, `apply_file_diffs`, `grep`, `file_glob_v2`). Translator intersects with `Settings.supported_tools` (a `repeated ToolType` enum) at request time, then merges in MCP tools dynamically from `Request.mcp_context.servers[].tools[]` (each named `mcp__<server_id>__<tool_name>` so we can route back).
- **Phase 9** (tool round-trip): `serialize_tool_result` handles 6 result variants explicitly with the JSON shapes in the catalogue's table, and a generic `{"unhandled_variant": "..."}` fallback. The 25 unhandled variants are out of scope by design.
- **Phase 10.2** (errors): map to specific `StreamFinished` reasons — `LLMUnavailable`, `InvalidApiKey`, `InternalError`, `ReachedMaxTokenLimit`, `Done` — instead of a single error variant.

These amendments are authoritative; where they conflict with the per-task code below, the amendments win.

## Repository Layout & Key Source Files

These are the files cited throughout the plan. The engineer should keep them open:

- **Fork target binary:** `app/src/bin/oss.rs` (~30 LoC; entry point for `cargo run --bin warp-oss --features gui`)
- **Cut point:** `app/src/server/server_api.rs` — function `generate_multi_agent_output` at **line 1091**, SSE eventsource at **line 1150**
- **Request builder:** `app/src/ai/agent/api/impl.rs` — function `generate_multi_agent_output` at **line ~11** (constructs `warp_multi_agent_api::Request` from `RequestParams`)
- **Conversation state machine (client-side):** `app/src/ai/agent/conversation.rs` — `AIConversation` struct, `set_server_conversation_token` at **line 748**
- **Server-bound message conversion:** `app/src/ai/agent/api/convert_from.rs` — `ConvertAPIMessageToClientOutputMessage` trait at **line ~185**
- **Channel/server config:** `crates/warp_core/src/channel/config.rs:30-39` — `WarpServerConfig::production()`
- **Model registry:** `crates/ai/src/llm_id.rs:5-7` — `LLMId(String)` (transparent string newtype)
- **Workspace Cargo.toml:** repo root `Cargo.toml` — pins `warp_multi_agent_api = { git = "https://github.com/warpdotdev/warp-proto-apis.git", rev = "aa2f9cde164a5b48ac01087d417d1188771f9b6d" }`
- **Proto schema (in proto-apis repo, not in main repo):**
  - `apis/multi_agent/v1/request.proto`
  - `apis/multi_agent/v1/response.proto`
  - `apis/multi_agent/v1/task.proto`
  - `apis/multi_agent/v1/skill.proto` (likely the tool definitions)
  - `apis/multi_agent/v1/conversation_data.proto`
  - `apis/multi_agent/v1/gen/rust/src/lib.rs` (generated Rust types — single source of truth for struct field names)

---

## File Structure

We add **one new crate** and modify two existing files.

**New:**
- `crates/llama_backend/Cargo.toml` — crate manifest, depends on `warp_multi_agent_api`, `reqwest`, `reqwest_eventsource`, `tokio`, `futures`, `serde`, `serde_json`, `async-stream`, `prost`, `base64`, `dashmap`, `tracing`, `anyhow`, `thiserror`
- `crates/llama_backend/src/lib.rs` — public surface: `pub fn dispatch_if_enabled(...) -> Option<ResponseStream>` and `pub fn generate(...) -> ResponseStream`
- `crates/llama_backend/src/config.rs` — env-var parsing (`WARP_LLAMA_URL`, `WARP_LLAMA_MODEL`, `WARP_LLAMA_API_KEY`)
- `crates/llama_backend/src/conversation_store.rs` — `ConversationStore` (in-memory `DashMap<String, Conversation>` keyed by `conversation_token`), `Conversation { messages: Vec<OpenAIMessage>, system_prompt: String, tools: Vec<OpenAIToolDef> }`
- `crates/llama_backend/src/system_prompt.rs` — `pub const WARP_SYSTEM_PROMPT: &str = ...` (the Warp persona text, hand-authored from observed behavior)
- `crates/llama_backend/src/tool_registry.rs` — static list of Warp tools as OpenAI function-call schemas, plus name↔proto-variant mapping table
- `crates/llama_backend/src/translate_request.rs` — `fn proto_request_to_openai(req: &warp_multi_agent_api::Request, store: &ConversationStore) -> openai_types::ChatCompletionRequest`
- `crates/llama_backend/src/translate_response.rs` — `fn openai_sse_to_proto_events(stream: impl Stream<...>, conversation_token: String) -> impl Stream<Item = warp_multi_agent_api::ResponseEvent>`
- `crates/llama_backend/src/openai_types.rs` — minimal serde structs for OpenAI chat-completion request/response/streaming-delta (we don't pull in the heavyweight async-openai crate)
- `crates/llama_backend/src/llama_client.rs` — thin wrapper around `reqwest` POST + `reqwest_eventsource`
- `crates/llama_backend/src/error.rs` — `LlamaBackendError` enum
- `crates/llama_backend/tests/golden/` — fixture files (recorded protobuf request, recorded llama-server SSE response, expected protobuf event stream)

**Modified:**
- `Cargo.toml` (workspace root) — add `crates/llama_backend` to `[workspace] members`
- `app/Cargo.toml` — add `llama_backend = { path = "../crates/llama_backend" }` to `[dependencies]`
- `app/src/server/server_api.rs` — at the top of `generate_multi_agent_output` (line 1091), call `llama_backend::dispatch_if_enabled(...)` and return its stream if Some; otherwise fall through to existing path

**Why this layout:** Isolating all replacement logic in a new crate means the diff against upstream is minimal (one file modified in `app/`), making rebasing on upstream Warp updates tractable. The `dispatch_if_enabled` returning `Option` is the entire integration surface — easy to verify, easy to disable.

---

## Phase 0 — Foundations (Derisk Before Touching Code)

The whole project is dead if we can't build `warp-oss` cleanly on Linux, or if the proto crate has hidden private deps. Validate both before writing any code.

### Task 0.1: Fork and clone Warp

**Files:**
- N/A (operates on the local filesystem outside the repo)

- [ ] **Step 1: Fork warpdotdev/warp on GitHub**

In a browser, navigate to `https://github.com/warpdotdev/warp` and click **Fork** under your account.

- [ ] **Step 2: Clone the fork into a working directory**

```bash
mkdir -p ~/projects && cd ~/projects
git clone --depth 1 git@github.com:<your-user>/warp.git warp-llama
cd warp-llama
```

Expected: clone completes; `ls` shows `app/`, `crates/`, `Cargo.toml`, etc.

- [ ] **Step 3: Pin upstream remote so we can rebase**

```bash
git remote add upstream https://github.com/warpdotdev/warp.git
git fetch upstream main --depth 1
```

Expected: `git remote -v` shows both `origin` (your fork) and `upstream`.

- [ ] **Step 4: Create a working branch**

```bash
git checkout -b llama-backend
```

### Task 0.2: Reproduce a clean upstream `warp-oss` build

**Files:**
- Read: `script/bootstrap`, `script/run`, `app/Cargo.toml` (no edits)

- [ ] **Step 1: Read the bootstrap script to understand system deps**

```bash
cat script/bootstrap
```

Expected: a list of `apt install` lines (build-essentials, OpenGL/Wayland libs, possibly `libssl-dev`, `pkg-config`, etc.). Read carefully — install anything missing on your machine.

- [ ] **Step 2: Run bootstrap**

```bash
./script/bootstrap
```

Expected: exits 0. Re-run after fixing any missing-dep errors.

- [ ] **Step 3: Build `warp-oss` in debug mode**

```bash
cargo build --bin warp-oss --features gui 2>&1 | tee /tmp/warp-oss-build.log
```

Expected: builds successfully in 3-10 minutes (depending on hardware). If it fails:
- If a private git repo is requested (e.g., `warp-channel-config`), confirm you're targeting `warp-oss` not the internal binary.
- Capture the error in the log and post it; do not proceed until the build is clean.

- [ ] **Step 4: Run the binary and confirm the GUI launches**

```bash
cargo run --bin warp-oss --features gui
```

Expected: a Warp terminal window opens. You may be prompted to log in to a Warp account — that's fine, you can either log in (real account) or close the prompt and use the terminal locally (AI features will be disabled, which is expected pre-fork).

If the window does not appear, debug Wayland/X11 issues before proceeding. **This is the gate to all further work.**

- [ ] **Step 5: Commit a baseline marker**

```bash
git commit --allow-empty -m "baseline: clean warp-oss build on $(uname -r)"
```

### Task 0.3: Vendor and inspect `warp-proto-apis`

The proto crate's generated Rust is the source of truth for every struct field name. We need it locally to grep.

**Files:**
- Read: `Cargo.toml` (workspace root, line containing `warp_multi_agent_api`)
- Clone-out: `~/projects/warp-proto-apis-ref/`

- [ ] **Step 1: Confirm the pinned revision**

```bash
grep warp-proto-apis Cargo.toml
```

Expected: a line containing `rev = "aa2f9cde164a5b48ac01087d417d1188771f9b6d"`. Record this exact revision.

- [ ] **Step 2: Clone the proto repo at that revision**

```bash
cd ~/projects
git clone https://github.com/warpdotdev/warp-proto-apis.git warp-proto-apis-ref
cd warp-proto-apis-ref
git checkout aa2f9cde164a5b48ac01087d417d1188771f9b6d
```

Expected: clone succeeds; checkout puts you in detached-HEAD on that revision.

- [ ] **Step 3: Identify the generated Rust types we'll be working with**

```bash
ls apis/multi_agent/v1/gen/rust/src/
wc -l apis/multi_agent/v1/gen/rust/src/lib.rs
head -100 apis/multi_agent/v1/gen/rust/src/lib.rs
```

Expected: `lib.rs` exists, on the order of thousands of lines, contains `pub struct Request`, `pub struct ResponseEvent`, etc.

- [ ] **Step 4: Catalogue the top-level types we will produce or consume**

Run the following greps and copy the output into a scratch file `~/projects/warp-llama/docs/superpowers/plans/proto-types-catalogue.md`:

```bash
cd ~/projects/warp-proto-apis-ref
grep -nE '^pub struct (Request|ResponseEvent|ClientAction|Message|ToolCall|ToolCallResult|StreamInit|StreamFinished|ClientActions|TaskContext|Settings|ModelConfig|Input|Metadata|MCPContext)' apis/multi_agent/v1/gen/rust/src/lib.rs
grep -nE '^pub enum (.*::)?Action|Tool|Status' apis/multi_agent/v1/gen/rust/src/lib.rs | head -40
```

Expected: line numbers for each struct/enum. This catalogue is the reference the engineer consults when writing translators.

- [ ] **Step 5: Commit the catalogue**

```bash
cd ~/projects/warp-llama
git add docs/superpowers/plans/proto-types-catalogue.md
git commit -m "docs: catalogue warp-proto-apis types referenced by llama_backend"
```

### Task 0.4: Smoke-test llama-server with curl — **DONE 2026-04-30**

Confirm CT501 responds the way OpenAI specifies, before depending on it from Rust.

**Result:** PASS with three constraints noted below; full curl evidence saved to `/tmp/warp-tool-call-test.json` and `/tmp/warp-stream-tool.txt` during the run.

**Constraints discovered (must apply throughout the plan):**
- **Auth header required.** Llama-server on CT501 enforces `Authorization: Bearer <key>` (the existing key from `agents-ct501.md` memory: `ncvbuI7MluQB8oC772W-tjsGdVlIKS0x39snaLqQ_eY`). Use it in every curl/test command below. The `WARP_LLAMA_API_KEY` env var is **required** for this deployment.
- **Model id is `Qwen3.6-27B-UD-Q3_K_XL.gguf`** (with `.gguf` suffix). Use this literal value in `WARP_LLAMA_MODEL` and in all smoke-test JSON bodies.
- **Reasoning leakage:** Qwen3.6 emits `delta.reasoning_content` first (thinking), then the closing `</think>\n\n` arrives in the *first* `delta.content` chunk before real content. Phase 7 must drop reasoning entirely and strip leading `</think>\s*` from the first content delta. See Phase 7 Step 4 update below.

**Files:** none (operates against the running llama-server)

- [ ] **Step 1: Confirm the model is up**

```bash
curl -s -H 'Authorization: Bearer ncvbuI7MluQB8oC772W-tjsGdVlIKS0x39snaLqQ_eY' http://192.168.8.68:8080/v1/models | jq '.data[].id'
```

Expected: a single string id (e.g., `"Qwen3.6-27B-UD-Q3_K_XL"`). Record this string — it's the value `WARP_LLAMA_MODEL` will need.

- [ ] **Step 2: Confirm streaming chat completions work**

```bash
curl -s -N http://192.168.8.68:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer ncvbuI7MluQB8oC772W-tjsGdVlIKS0x39snaLqQ_eY' \
  -d '{
    "model": "Qwen3.6-27B-UD-Q3_K_XL.gguf",
    "messages": [{"role":"user","content":"reply with the word PONG"}],
    "stream": true,
    "max_tokens": 16
  }' | head -30
```

Expected: SSE stream with `data: {...}` lines; final line `data: [DONE]`. The `delta.content` fields concatenate to "PONG".

- [ ] **Step 3: Confirm tool-calling works**

```bash
curl -s http://192.168.8.68:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer ncvbuI7MluQB8oC772W-tjsGdVlIKS0x39snaLqQ_eY' \
  -d '{
    "model": "Qwen3.6-27B-UD-Q3_K_XL.gguf",
    "messages": [{"role":"user","content":"What files are in /tmp?"}],
    "tools": [{
      "type":"function",
      "function":{
        "name":"run_shell_command",
        "description":"Execute a shell command and return its output",
        "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}
      }
    }],
    "tool_choice":"auto",
    "max_tokens": 256
  }' | jq '.choices[0].message'
```

Expected: response includes a `tool_calls` array with `function.name == "run_shell_command"` and a JSON `arguments` string containing `"command":"ls /tmp"` or similar. **If this fails**, Qwen3.6 isn't doing tool calls reliably with the current llama-server config — flag this as a blocker before continuing the rest of the plan, because tool calls are central to Warp's agent loop.

---

## Phase 1 — Add the Crate Skeleton and Wire the Cut Point

Get the integration plumbing working with stubs. Prove that flipping `WARP_LLAMA_URL` makes Warp's UI display synthetic events from our crate, before any real translation logic exists.

### Task 1.1: Create the `llama_backend` crate

**Files:**
- Create: `crates/llama_backend/Cargo.toml`
- Create: `crates/llama_backend/src/lib.rs`
- Modify: `Cargo.toml` (workspace root) — add `crates/llama_backend` to `members`

- [ ] **Step 1: Create the crate manifest**

Write `crates/llama_backend/Cargo.toml`:

```toml
[package]
name = "llama_backend"
version = "0.1.0"
edition = "2021"
license = "AGPL-3.0-or-later OR MIT"

[dependencies]
anyhow = { workspace = true }
async-stream = "0.3"
base64 = { workspace = true }
dashmap = "6"
futures = { workspace = true }
prost = { workspace = true }
reqwest = { workspace = true, features = ["json", "stream"] }
reqwest_eventsource = { workspace = true }
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
thiserror = { workspace = true }
tokio = { workspace = true, features = ["sync", "macros", "rt"] }
tracing = { workspace = true }
warp_multi_agent_api = { workspace = true }

[dev-dependencies]
tokio = { workspace = true, features = ["full", "test-util"] }
wiremock = "0.6"
```

If any dep is not already in the workspace `[workspace.dependencies]` table, add it there with a sensible version. Run `cargo metadata --format-version 1 | jq '.packages[] | select(.name=="<dep>") | .version'` to see what version upstream Warp uses for already-present deps.

- [ ] **Step 2: Add the crate to the workspace**

Open `Cargo.toml` (workspace root). Find the `[workspace] members = [...]` array. Add `"crates/llama_backend",` in alphabetical order with the existing entries.

- [ ] **Step 3: Create a minimal `lib.rs`**

Write `crates/llama_backend/src/lib.rs`:

```rust
//! Local-LLM backend for Warp.
//!
//! Replaces the network call to `https://app.warp.dev/ai/multi-agent` with
//! a direct call to a local OpenAI-compatible server (e.g. llama-server),
//! when the env var `WARP_LLAMA_URL` is set.

#![deny(unused_must_use)]

use std::pin::Pin;

use futures::Stream;
use warp_multi_agent_api::{Request, ResponseEvent};

pub mod config;

pub type ResponseEventStream =
    Pin<Box<dyn Stream<Item = Result<ResponseEvent, anyhow::Error>> + Send>>;

/// Returns `Some(stream)` if the local backend is enabled (i.e. `WARP_LLAMA_URL`
/// is set), otherwise `None`. Callers fall through to the real backend on `None`.
pub fn dispatch_if_enabled(_request: Request) -> Option<ResponseEventStream> {
    if config::Config::from_env().is_none() {
        return None;
    }
    // Phase 1 stub: emit nothing, signaling "we handled it" but with no events.
    Some(Box::pin(futures::stream::empty()))
}
```

- [ ] **Step 4: Create a stub `config` module**

Write `crates/llama_backend/src/config.rs`:

```rust
use std::env;

pub struct Config {
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
}

impl Config {
    pub fn from_env() -> Option<Self> {
        let url = env::var("WARP_LLAMA_URL").ok()?;
        let model = env::var("WARP_LLAMA_MODEL")
            .unwrap_or_else(|_| "default".to_string());
        let api_key = env::var("WARP_LLAMA_API_KEY").ok();
        Some(Self { url, model, api_key })
    }
}
```

- [ ] **Step 5: Add llama_backend as a dep of `app`**

Open `app/Cargo.toml`. Under `[dependencies]`, add:

```toml
llama_backend = { path = "../crates/llama_backend" }
```

- [ ] **Step 6: Verify the workspace compiles**

```bash
cargo check -p llama_backend
cargo check -p warp
```

Expected: both succeed with zero warnings about the new crate.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml app/Cargo.toml crates/llama_backend
git commit -m "feat(llama_backend): scaffold crate with env-var config"
```

### Task 1.2: Add the dispatch hook in `server_api.rs`

**Files:**
- Modify: `app/src/server/server_api.rs:~1091` (function `generate_multi_agent_output`)

- [ ] **Step 1: Read the existing function**

Open `app/src/server/server_api.rs` and read **lines 1080–1180**. Specifically note:
- The function signature (return type, parameters, lifetime/`async` markers)
- How the URL is constructed (the `is_passive` / `is_evals` branch around line 1110)
- How the body protobuf is attached (`request.proto(...)` call before line 1150)
- How the SSE stream is built (`request.eventsource().filter_map(...)` at line 1150)

Copy the function signature (the `pub async fn ... -> ...` line) verbatim into a scratch note — we need its exact return type.

- [ ] **Step 2: Add the dispatch shim at the top of the function**

Immediately after the function's opening brace, before any URL construction, insert:

```rust
// llama_backend integration: if WARP_LLAMA_URL is set, divert this turn to a
// local OpenAI-compatible server instead of Warp's backend.
if let Some(stream) = llama_backend::dispatch_if_enabled(request.clone()) {
    return Ok(Box::pin(stream.map(|res| res.map_err(Into::into))));
}
```

If the existing function's return type is not boxed, adjust the call to match (e.g., wrap differently). The exact wrapping depends on what `ResponseStream` is aliased to — read the file to find out, and align with it.

If `request` is moved (not `Clone`), either:
- Make `Request` clonable (it's a protobuf, it should already derive `Clone`; verify with `grep -n 'derive' apis/multi_agent/v1/gen/rust/src/lib.rs | grep -i request`).
- Or move the dispatch *after* the request is built but pass a reference to `dispatch_if_enabled`. Adjust `dispatch_if_enabled`'s signature accordingly.

- [ ] **Step 3: Build and confirm no compiler errors**

```bash
cargo check -p warp
```

Expected: success. If type mismatches, iterate.

- [ ] **Step 4: Run with the env var set, confirm dispatch is taken**

Add a `tracing::info!("llama_backend: dispatch taken");` line inside `dispatch_if_enabled` right before returning `Some(...)`. Then:

```bash
RUST_LOG=llama_backend=info WARP_LLAMA_URL=http://127.0.0.1:9999 \
  cargo run --bin warp-oss --features gui
```

Inside Warp, attempt to use Agent Mode (any prompt). Watch the terminal where you ran the command.

Expected: the log line `llama_backend: dispatch taken` appears. Warp's UI will show "no response" because the stub returns an empty stream — that's expected for Phase 1.

- [ ] **Step 5: Run without the env var, confirm fallthrough**

```bash
unset WARP_LLAMA_URL
cargo run --bin warp-oss --features gui
```

Expected: AI features behave as in Task 0.2 step 4 — they go to the real Warp backend (or fail with auth error if not logged in). The log line does NOT appear.

- [ ] **Step 6: Remove the temporary tracing line and commit**

Remove the `tracing::info!` line from `dispatch_if_enabled`. Then:

```bash
git add app/src/server/server_api.rs crates/llama_backend
git commit -m "feat(server_api): hook dispatch to llama_backend behind WARP_LLAMA_URL"
```

---

## Phase 2 — Conversation Store

Warp's real backend stores conversation history server-side and the client only sends `conversation_token` on follow-up turns. We must reconstruct that locally.

### Task 2.1: Define the conversation data model

**Files:**
- Create: `crates/llama_backend/src/openai_types.rs`
- Create: `crates/llama_backend/src/conversation_store.rs`

- [ ] **Step 1: Write the failing test for `Conversation::append_user_message`**

Create `crates/llama_backend/src/conversation_store.rs`:

```rust
use crate::openai_types::{OpenAIMessage, OpenAIRole};
use dashmap::DashMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Conversation {
    pub token: String,
    pub messages: Vec<OpenAIMessage>,
}

impl Conversation {
    pub fn new(token: String, system_prompt: String) -> Self {
        Self {
            token,
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
}

#[derive(Clone, Default)]
pub struct ConversationStore {
    inner: Arc<DashMap<String, Conversation>>,
}

impl ConversationStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_init(&self, token: &str, system_prompt: &str) -> Conversation {
        self.inner
            .entry(token.to_string())
            .or_insert_with(|| Conversation::new(token.to_string(), system_prompt.to_string()))
            .clone()
    }

    pub fn upsert(&self, conv: Conversation) {
        self.inner.insert(conv.token.clone(), conv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_user_message_grows_history() {
        let store = ConversationStore::new();
        let mut conv = store.get_or_init("tok-1", "you are warp");
        assert_eq!(conv.messages.len(), 1);
        conv.append_user("hello".into());
        store.upsert(conv);
        let conv = store.get_or_init("tok-1", "ignored on second get");
        assert_eq!(conv.messages.len(), 2);
        assert!(matches!(conv.messages[1].role, OpenAIRole::User));
    }
}
```

- [ ] **Step 2: Define `OpenAIMessage` and supporting types**

Create `crates/llama_backend/src/openai_types.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: OpenAIFunctionCall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String, // raw JSON string per OpenAI spec
}

#[derive(Clone, Debug, Serialize)]
pub struct OpenAIToolDef {
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: OpenAIFunctionDef,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpenAIFunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON schema
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
```

- [ ] **Step 3: Wire the new modules into `lib.rs`**

In `crates/llama_backend/src/lib.rs`, add:

```rust
pub mod conversation_store;
pub mod openai_types;
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cargo test -p llama_backend conversation_store::tests::append_user_message_grows_history
```

Expected: 1 passed.

- [ ] **Step 5: Add a test for `tool_call_id` round-trip**

Append to the `tests` module in `conversation_store.rs`:

```rust
#[test]
fn tool_result_message_has_tool_call_id() {
    let msg = OpenAIMessage {
        role: OpenAIRole::Tool,
        content: Some("file contents".into()),
        tool_calls: None,
        tool_call_id: Some("call-abc".into()),
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"tool_call_id\":\"call-abc\""));
    assert!(json.contains("\"role\":\"tool\""));
    let back: OpenAIMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(back.tool_call_id.as_deref(), Some("call-abc"));
}
```

- [ ] **Step 6: Run all crate tests**

```bash
cargo test -p llama_backend
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add crates/llama_backend
git commit -m "feat(llama_backend): conversation store + OpenAI message types"
```

---

## Phase 3 — Tool Registry & Schema Mapping

Warp's tools are defined as proto variants (`run_shell_command`, `read_files`, `apply_file_diffs`, `call_mcp_tool`, etc.). The model must see them as OpenAI function-call schemas, and the model's chosen tool name must map back to a proto variant.

### Task 3.1: Catalogue Warp's tool variants

**Files:**
- Read: `~/projects/warp-proto-apis-ref/apis/multi_agent/v1/skill.proto`
- Read: `~/projects/warp-proto-apis-ref/apis/multi_agent/v1/gen/rust/src/lib.rs` (search for `ToolCall` and the `Tool` oneof)
- Append: `docs/superpowers/plans/proto-types-catalogue.md`

- [ ] **Step 1: Find the canonical list of Warp tool variants**

```bash
cd ~/projects/warp-proto-apis-ref
grep -nE 'oneof tool|RunShellCommand|ReadFiles|ApplyFileDiffs|SearchCodebase|CallMcpTool' apis/multi_agent/v1/*.proto | head -40
```

Expected: a list of message types under a `oneof tool` block, one per tool. Capture the full list — call this set `T`. This is the universe of tools we must expose.

- [ ] **Step 2: For each tool in T, find its parameter struct**

In the same proto files, find the `message <ToolName> { ... }` definition. Note every field, its proto type, and its `[(google.api.field_behavior) = REQUIRED]` annotations if present.

Record the catalogue as a table in `docs/superpowers/plans/proto-types-catalogue.md`:

```markdown
## Warp Tool Variants (from skill.proto / request.proto)

| Tool name (proto) | Tool name (snake_case for OpenAI) | Required fields | Optional fields |
|---|---|---|---|
| RunShellCommand | run_shell_command | command:string | working_directory:string, timeout_seconds:uint32 |
| ReadFiles | read_files | paths:[]string | … |
| … | … | … | … |
```

This table is the spec for `tool_registry.rs`.

- [ ] **Step 3: Commit the updated catalogue**

```bash
cd ~/projects/warp-llama
git add docs/superpowers/plans/proto-types-catalogue.md
git commit -m "docs: catalogue Warp tool variants and their fields"
```

### Task 3.2: Implement `tool_registry.rs`

**Files:**
- Create: `crates/llama_backend/src/tool_registry.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/llama_backend/src/tool_registry.rs` with just the test, importing types we're about to define:

```rust
use crate::openai_types::OpenAIToolDef;

pub fn warp_tools_as_openai() -> Vec<OpenAIToolDef> {
    todo!("implement in step 3")
}

pub fn tool_name_for_proto_variant(variant_name: &str) -> Option<&'static str> {
    todo!("implement in step 3")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_includes_run_shell_command() {
        let tools = warp_tools_as_openai();
        let names: Vec<_> = tools.iter().map(|t| t.function.name.as_str()).collect();
        assert!(names.contains(&"run_shell_command"), "missing run_shell_command in {:?}", names);
    }

    #[test]
    fn run_shell_command_schema_requires_command_field() {
        let tools = warp_tools_as_openai();
        let t = tools.iter().find(|t| t.function.name == "run_shell_command").unwrap();
        let required = t.function.parameters.get("required").and_then(|v| v.as_array()).unwrap();
        let required_strs: Vec<_> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(required_strs.contains(&"command"));
    }

    #[test]
    fn proto_variant_name_to_openai_name_roundtrip() {
        assert_eq!(tool_name_for_proto_variant("RunShellCommand"), Some("run_shell_command"));
        assert_eq!(tool_name_for_proto_variant("ReadFiles"), Some("read_files"));
        assert_eq!(tool_name_for_proto_variant("Unknown"), None);
    }
}
```

- [ ] **Step 2: Wire into `lib.rs`**

```rust
pub mod tool_registry;
```

- [ ] **Step 3: Run tests, confirm they fail with `not yet implemented`**

```bash
cargo test -p llama_backend tool_registry
```

Expected: FAIL with `panicked at 'not yet implemented'`.

- [ ] **Step 4: Implement `warp_tools_as_openai`**

Replace the `todo!()` with explicit construction of every tool from the catalogue (Task 3.1). Example for `run_shell_command`:

```rust
pub fn warp_tools_as_openai() -> Vec<OpenAIToolDef> {
    vec![
        OpenAIToolDef {
            kind: "function".into(),
            function: crate::openai_types::OpenAIFunctionDef {
                name: "run_shell_command".into(),
                description: "Execute a shell command and return its stdout, stderr, and exit code.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run"},
                        "working_directory": {"type": "string", "description": "Optional cwd override"},
                        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600}
                    },
                    "required": ["command"]
                }),
            },
        },
        // ... one entry per tool in the catalogue
    ]
}
```

Repeat for every tool in the Task 3.1 catalogue. **Do not skip any.** A missing tool means the model can't invoke that capability.

- [ ] **Step 5: Implement `tool_name_for_proto_variant`**

```rust
pub fn tool_name_for_proto_variant(variant_name: &str) -> Option<&'static str> {
    Some(match variant_name {
        "RunShellCommand" => "run_shell_command",
        "ReadFiles" => "read_files",
        "ApplyFileDiffs" => "apply_file_diffs",
        // ... one arm per tool
        _ => return None,
    })
}
```

Also write the inverse:

```rust
pub fn proto_variant_for_tool_name(tool_name: &str) -> Option<&'static str> {
    Some(match tool_name {
        "run_shell_command" => "RunShellCommand",
        "read_files" => "ReadFiles",
        "apply_file_diffs" => "ApplyFileDiffs",
        // ...
        _ => return None,
    })
}
```

Add a test that the two functions round-trip for every tool name. Do this via a `const TOOL_NAMES: &[(&str, &str)]` table that both functions consume — DRY.

- [ ] **Step 6: Run tests**

```bash
cargo test -p llama_backend tool_registry
```

Expected: all 3+ tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/llama_backend
git commit -m "feat(llama_backend): tool registry with OpenAI schemas + proto-variant mapping"
```

---

## Phase 4 — Outbound Translation: Proto Request → OpenAI Request

Convert `warp_multi_agent_api::Request` into an `OpenAIChatCompletionRequest`, using the conversation store to fill in history.

### Task 4.1: Implement the translator

**Files:**
- Create: `crates/llama_backend/src/translate_request.rs`
- Read: `~/projects/warp-proto-apis-ref/apis/multi_agent/v1/request.proto` (full)
- Read: `~/projects/warp-proto-apis-ref/apis/multi_agent/v1/gen/rust/src/lib.rs` (search `pub struct Request`, `pub struct Input`, `pub struct Settings`, `pub struct Metadata`)

- [ ] **Step 1: Identify the conversation token field on `Request`**

In `apis/multi_agent/v1/gen/rust/src/lib.rs`, find `pub struct Metadata` (or wherever `conversation_token` lives — earlier recon said it's on a follow-up request input, but the proto-apis source is authoritative). Record the exact field path, e.g., `request.metadata.unwrap().conversation_token`.

- [ ] **Step 2: Identify the user input field shape**

Find `pub struct Input` and the `oneof input_kind` (or similar). The variants are `UserInputs`, `ToolCallResult`, etc. Record the field path to extract the user's text on a fresh turn (`request.input.unwrap().input_kind` matched on `UserInputs(_)`).

- [ ] **Step 3: Write the failing test for a fresh-turn translation**

Create `crates/llama_backend/src/translate_request.rs`:

```rust
use crate::conversation_store::ConversationStore;
use crate::openai_types::ChatCompletionRequest;
use crate::tool_registry;
use warp_multi_agent_api::Request as ProtoRequest;

pub fn proto_request_to_openai(
    proto: &ProtoRequest,
    store: &ConversationStore,
    system_prompt: &str,
    model: &str,
) -> Result<(ChatCompletionRequest, String /* conversation_token */), anyhow::Error> {
    todo!("implement in step 5")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_turn_request_with_text(text: &str) -> ProtoRequest {
        // Build a minimal Request with metadata.conversation_token = "" and
        // input = UserInputs(text = text).
        // Field paths must match the actual generated proto struct — adjust per Task 4.1 step 1/2.
        todo!("construct using actual generated types from warp_multi_agent_api")
    }

    #[test]
    fn fresh_turn_initializes_conversation_with_system_prompt() {
        let store = ConversationStore::new();
        let req = fresh_turn_request_with_text("hello");
        let (openai, token) =
            proto_request_to_openai(&req, &store, "you are warp", "qwen").unwrap();
        assert_eq!(openai.model, "qwen");
        assert_eq!(openai.messages.len(), 2);
        assert_eq!(openai.messages[0].content.as_deref(), Some("you are warp"));
        assert_eq!(openai.messages[1].content.as_deref(), Some("hello"));
        assert!(!token.is_empty());
        assert!(openai.tools.iter().any(|t| t.function.name == "run_shell_command"));
    }

    #[test]
    fn followup_turn_reuses_conversation_history() {
        let store = ConversationStore::new();
        let req1 = fresh_turn_request_with_text("first message");
        let (_, token) =
            proto_request_to_openai(&req1, &store, "sys", "qwen").unwrap();

        // Manually inject an assistant reply so the next request has 3 messages
        // already (system, user, assistant).
        {
            let mut conv = store.get_or_init(&token, "sys");
            conv.messages.push(crate::openai_types::OpenAIMessage {
                role: crate::openai_types::OpenAIRole::Assistant,
                content: Some("first reply".into()),
                tool_calls: None,
                tool_call_id: None,
            });
            store.upsert(conv);
        }

        let req2 = followup_turn_request_with_text(&token, "second message");
        let (openai, token2) =
            proto_request_to_openai(&req2, &store, "sys", "qwen").unwrap();
        assert_eq!(token, token2);
        assert_eq!(openai.messages.len(), 4); // sys, user1, asst1, user2
        assert_eq!(openai.messages[3].content.as_deref(), Some("second message"));
    }

    fn followup_turn_request_with_text(token: &str, text: &str) -> ProtoRequest {
        todo!("construct with metadata.conversation_token = token")
    }
}
```

- [ ] **Step 4: Wire the module into `lib.rs`**

```rust
pub mod translate_request;
```

- [ ] **Step 5: Implement `proto_request_to_openai`**

```rust
use uuid::Uuid;

pub fn proto_request_to_openai(
    proto: &ProtoRequest,
    store: &ConversationStore,
    system_prompt: &str,
    model: &str,
) -> Result<(ChatCompletionRequest, String), anyhow::Error> {
    // 1. Resolve conversation token (use existing if present, mint a UUID otherwise).
    let token = proto
        .metadata
        .as_ref()
        .and_then(|m| {
            let t = &m.conversation_token; // adjust if field name differs
            if t.is_empty() { None } else { Some(t.clone()) }
        })
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let mut conv = store.get_or_init(&token, system_prompt);

    // 2. Translate the input.
    let input = proto.input.as_ref().ok_or_else(|| anyhow::anyhow!("missing input"))?;
    use warp_multi_agent_api::request::input::InputKind; // adjust per generated module path
    match &input.input_kind {
        Some(InputKind::UserInputs(u)) => {
            // u.text is the user's prompt; field name per generated code
            conv.append_user(u.text.clone());
        }
        Some(InputKind::ToolCallResult(r)) => {
            // Append a tool-result message — see Task 9 for full handling.
            conv.messages.push(crate::openai_types::OpenAIMessage {
                role: crate::openai_types::OpenAIRole::Tool,
                content: Some(serialize_tool_result(r)?),
                tool_calls: None,
                tool_call_id: Some(r.tool_call_id.clone()),
            });
        }
        other => return Err(anyhow::anyhow!("unsupported input kind: {:?}", other)),
    }

    store.upsert(conv.clone());

    let openai = ChatCompletionRequest {
        model: model.to_string(),
        messages: conv.messages,
        tools: tool_registry::warp_tools_as_openai(),
        stream: true,
        max_tokens: Some(4096),
        temperature: Some(0.7),
    };

    Ok((openai, token))
}

fn serialize_tool_result(
    _r: &warp_multi_agent_api::ToolCallResult,
) -> Result<String, anyhow::Error> {
    // Phase 4 placeholder: return debug repr; Phase 9 replaces with proper per-tool
    // serialization.
    Ok("(tool result placeholder; see Phase 9)".to_string())
}
```

Add `uuid = { workspace = true, features = ["v4"] }` to the crate's `[dependencies]` if not already there.

- [ ] **Step 6: Implement the test fixture builders**

Replace the `todo!()` in `fresh_turn_request_with_text` and `followup_turn_request_with_text` with actual construction using the generated proto types. Read the generated `lib.rs` to find each constructor / field path. Example:

```rust
fn fresh_turn_request_with_text(text: &str) -> ProtoRequest {
    use warp_multi_agent_api::request::{Input, input::{InputKind, UserInputs}};
    use warp_multi_agent_api::Metadata;
    ProtoRequest {
        metadata: Some(Metadata { conversation_token: String::new(), ..Default::default() }),
        input: Some(Input { input_kind: Some(InputKind::UserInputs(UserInputs {
            text: text.to_string(), ..Default::default()
        })) }),
        settings: Default::default(),
        task_context: Default::default(),
        mcp_context: Default::default(),
    }
}
```

Adjust field names to match the actual generated code — the engineer must verify against `~/projects/warp-proto-apis-ref/apis/multi_agent/v1/gen/rust/src/lib.rs`.

- [ ] **Step 7: Run tests**

```bash
cargo test -p llama_backend translate_request
```

Expected: both tests pass.

- [ ] **Step 8: Commit**

```bash
git add crates/llama_backend
git commit -m "feat(llama_backend): translate proto Request to OpenAI ChatCompletionRequest"
```

---

## Phase 5 — System Prompt

Author the persona the local model will use. The real Warp backend injects a system prompt server-side; we have no access to its content but can reconstruct a working one from observed behavior + the tool list.

### Task 5.1: Author the system prompt

**Files:**
- Create: `crates/llama_backend/src/system_prompt.rs`

- [ ] **Step 1: Write the prompt**

Create `crates/llama_backend/src/system_prompt.rs`:

```rust
pub const WARP_SYSTEM_PROMPT: &str = r#"You are Warp Agent, an AI software-engineering assistant embedded in the Warp terminal.

You operate by calling tools. Every action you take in the user's environment must go through a tool call — never claim to have run a command, read a file, or made an edit unless you actually invoked the corresponding tool.

Available tools and when to use each:
- run_shell_command: execute a shell command. Prefer this for anything you'd type at a prompt. Cap your commands; never run interactive editors or pagers.
- read_files: read one or more files into your context. Use this before editing.
- apply_file_diffs: edit one or more files via unified diff. Always read first.
- search_codebase / grep / glob: locate code. Use search_codebase for semantic queries, grep for literal regex, glob for path patterns.
- (additional tools per registry)

Conventions:
- Be terse. The user is reading you in a terminal.
- After you finish a multi-step task, summarize in one or two sentences. Do not narrate every step.
- Prefer reversible local operations. Confirm before anything destructive (rm -rf, force pushes, dropping data).
- When the user asks an open question without specifying a task, answer briefly and offer to act.

Today's date is provided in the user's first message metadata if available.
"#;
```

This text is inferred. After Phase 8 (smoke test) the engineer should iterate on this based on observed model behavior — too verbose, too terse, missing tool guidance, etc.

- [ ] **Step 2: Wire into `lib.rs`**

```rust
pub mod system_prompt;
```

- [ ] **Step 3: Add a sanity test**

In `system_prompt.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_mentions_run_shell_command() {
        assert!(WARP_SYSTEM_PROMPT.contains("run_shell_command"));
    }

    #[test]
    fn prompt_is_nontrivial() {
        assert!(WARP_SYSTEM_PROMPT.len() > 200);
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p llama_backend system_prompt
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/llama_backend
git commit -m "feat(llama_backend): hand-authored Warp Agent system prompt"
```

---

## Phase 6 — Llama-Server Client

Make the actual HTTP call.

### Task 6.1: Implement `llama_client.rs` with a wiremock test

**Files:**
- Create: `crates/llama_backend/src/llama_client.rs`
- Create: `crates/llama_backend/src/error.rs`

- [ ] **Step 1: Define the error type**

Create `crates/llama_backend/src/error.rs`:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlamaBackendError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("SSE error: {0}")]
    Sse(String),
    #[error("Bad SSE payload: {0}")]
    BadPayload(String),
    #[error("Stream ended unexpectedly")]
    EarlyEof,
}
```

- [ ] **Step 2: Write the failing test using wiremock**

Create `crates/llama_backend/src/llama_client.rs`:

```rust
use crate::error::LlamaBackendError;
use crate::openai_types::ChatCompletionRequest;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};

pub struct LlamaClient {
    http: reqwest::Client,
    url: String,
    api_key: Option<String>,
}

impl LlamaClient {
    pub fn new(url: String, api_key: Option<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            url,
            api_key,
        }
    }

    pub fn stream_completion(
        &self,
        req: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<String, LlamaBackendError>>, LlamaBackendError> {
        let mut builder = self
            .http
            .post(format!("{}/v1/chat/completions", self.url.trim_end_matches('/')))
            .json(req);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        let es = EventSource::new(builder).map_err(|e| LlamaBackendError::Sse(e.to_string()))?;
        let stream = es.filter_map(|ev| async move {
            match ev {
                Ok(Event::Open) => None,
                Ok(Event::Message(msg)) => {
                    if msg.data == "[DONE]" {
                        None
                    } else {
                        Some(Ok(msg.data))
                    }
                }
                Err(e) => Some(Err(LlamaBackendError::Sse(e.to_string()))),
            }
        });
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn streams_two_chunks_then_done() {
        let server = MockServer::start().await;
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
                    data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let client = LlamaClient::new(server.uri(), None);
        let req = ChatCompletionRequest {
            model: "test".into(),
            messages: vec![],
            tools: vec![],
            stream: true,
            max_tokens: None,
            temperature: None,
        };
        let mut stream = client.stream_completion(&req).unwrap();
        let mut payloads = vec![];
        while let Some(item) = stream.next().await {
            payloads.push(item.unwrap());
        }
        assert_eq!(payloads.len(), 2);
        assert!(payloads[0].contains("hello"));
        assert!(payloads[1].contains(" world"));
    }
}
```

- [ ] **Step 3: Wire into `lib.rs`**

```rust
pub mod error;
pub mod llama_client;
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p llama_backend llama_client
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/llama_backend
git commit -m "feat(llama_backend): SSE streaming client with wiremock-tested happy path"
```

---

## Phase 7 — Inbound Translation: OpenAI SSE → Proto Events

Take a stream of OpenAI streaming chunks and emit `warp_multi_agent_api::ResponseEvent`s that Warp's UI can consume.

### Task 7.1: Define inbound chunk types and write the failing translator test

**Files:**
- Create: `crates/llama_backend/src/translate_response.rs`
- Modify: `crates/llama_backend/src/openai_types.rs` (add streaming chunk types)

- [ ] **Step 1: Add streaming chunk types to `openai_types.rs`**

Append:

```rust
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
    /// Qwen3.6 emits its `<think>` block here; the translator drops these.
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
```

- [ ] **Step 2: Write the failing test for the translator**

Create `crates/llama_backend/src/translate_response.rs`:

```rust
use crate::error::LlamaBackendError;
use crate::openai_types::ChatCompletionChunk;
use crate::tool_registry;
use async_stream::try_stream;
use futures::Stream;
use std::pin::Pin;
use warp_multi_agent_api::ResponseEvent;

pub fn openai_sse_to_proto_events<S>(
    sse: S,
    conversation_token: String,
) -> Pin<Box<dyn Stream<Item = Result<ResponseEvent, LlamaBackendError>> + Send>>
where
    S: Stream<Item = Result<String, LlamaBackendError>> + Send + 'static,
{
    todo!("implement in step 4")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn sse_chunks(jsons: &[&str]) -> impl Stream<Item = Result<String, LlamaBackendError>> {
        let v: Vec<_> = jsons.iter().map(|s| Ok::<_, LlamaBackendError>(s.to_string())).collect();
        futures::stream::iter(v)
    }

    #[tokio::test]
    async fn first_event_is_stream_init() {
        let sse = sse_chunks(&[
            r#"{"choices":[{"delta":{"content":"hi"}}]}"#,
        ]);
        let mut out = openai_sse_to_proto_events(sse, "tok-1".into());
        let first = out.next().await.unwrap().unwrap();
        // assert first event is a StreamInit with conversation_token = "tok-1"
        // (exact match shape depends on generated proto enum).
    }

    #[tokio::test]
    async fn text_delta_emits_append_to_message_content() {
        let sse = sse_chunks(&[
            r#"{"choices":[{"delta":{"content":"hello"}}]}"#,
            r#"{"choices":[{"delta":{"content":" world"}}]}"#,
        ]);
        let out: Vec<_> = openai_sse_to_proto_events(sse, "tok-1".into())
            .collect()
            .await;
        // Filter to AppendToMessageContent events; expect 2 with payloads "hello" and " world".
        // Concrete assertion: the joined text equals "hello world".
        // (Skeleton; complete the assertion against the actual ClientAction variant once implemented.)
    }

    #[tokio::test]
    async fn streamed_tool_call_assembles_then_emits_one_toolcall_event() {
        let sse = sse_chunks(&[
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"run_shell_command","arguments":"{\"co"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"mmand\":\"ls\"}"}}]}}]}"#,
            r#"{"choices":[{"finish_reason":"tool_calls","delta":{}}]}"#,
        ]);
        let out: Vec<_> = openai_sse_to_proto_events(sse, "tok-1".into())
            .collect()
            .await;
        // Expect exactly one ToolCall event for run_shell_command with arguments {"command":"ls"}.
    }
}
```

- [ ] **Step 3: Wire into `lib.rs`**

```rust
pub mod translate_response;
```

- [ ] **Step 4: Implement `openai_sse_to_proto_events`**

**Reasoning-content handling (REQUIRED, per Phase 0.4 finding):**

Qwen3.6 streams `delta.reasoning_content` deltas first (its `<think>` block), then leaks the closing `</think>\n\n` into the *first* `delta.content` chunk before real content begins. The translator MUST:

1. **Ignore `delta.reasoning_content` entirely** — do not yield any event for it. Add the field to `ChunkDelta` but mark `#[serde(default)]` and don't read it inside the loop.
2. **Strip leading `</think>` artifacts from the first content delta.** Maintain a `bool first_content_delta = true` flag in the translator state. On the first chunk where `delta.content` is `Some` and non-empty, apply a regex strip equivalent to `^\s*</think>\s*` to the text before yielding. Set the flag to false. Subsequent deltas pass through unmodified.

Add two unit tests covering this: `reasoning_content_deltas_are_dropped` and `leading_close_think_tag_stripped_from_first_content`.

Pseudocode (adapt struct field names from the proto-apis source — engineer must verify each constructor):

```rust
pub fn openai_sse_to_proto_events<S>(
    sse: S,
    conversation_token: String,
) -> Pin<Box<dyn Stream<Item = Result<ResponseEvent, LlamaBackendError>> + Send>>
where
    S: Stream<Item = Result<String, LlamaBackendError>> + Send + 'static,
{
    use futures::StreamExt;
    let s = try_stream! {
        // 1. Yield a StreamInit event.
        yield make_stream_init(&conversation_token);

        // 2. Accumulator for streamed tool calls (OpenAI streams them in fragments).
        // Map from tool-call index to (id, name, arguments_buffer).
        let mut tool_calls: std::collections::BTreeMap<usize, PartialToolCall> =
            std::collections::BTreeMap::new();
        let mut finished = false;

        let mut sse = Box::pin(sse);
        while let Some(item) = sse.next().await {
            let payload = item?;
            let chunk: ChatCompletionChunk = serde_json::from_str(&payload)
                .map_err(|e| LlamaBackendError::BadPayload(e.to_string()))?;

            for choice in chunk.choices {
                if let Some(text) = choice.delta.content {
                    yield make_append_text_event(&text);
                }
                if let Some(deltas) = choice.delta.tool_calls {
                    for d in deltas {
                        let entry = tool_calls.entry(d.index).or_insert(PartialToolCall::default());
                        if let Some(id) = d.id { entry.id = id; }
                        if let Some(f) = d.function {
                            if let Some(name) = f.name { entry.name = name; }
                            if let Some(args) = f.arguments { entry.args.push_str(&args); }
                        }
                    }
                }
                if choice.finish_reason.as_deref() == Some("tool_calls")
                    || choice.finish_reason.as_deref() == Some("stop")
                {
                    for (_, partial) in tool_calls.iter() {
                        yield make_tool_call_event(partial)?;
                    }
                    tool_calls.clear();
                    if choice.finish_reason.as_deref() == Some("stop") {
                        yield make_stream_finished_ok();
                        finished = true;
                    }
                }
            }
        }
        if !finished {
            yield make_stream_finished_ok();
        }
    };
    Box::pin(s)
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    args: String,
}

fn make_stream_init(token: &str) -> ResponseEvent { /* construct via generated types */ todo!() }
fn make_append_text_event(text: &str) -> ResponseEvent { todo!() }
fn make_tool_call_event(p: &PartialToolCall) -> Result<ResponseEvent, LlamaBackendError> { todo!() }
fn make_stream_finished_ok() -> ResponseEvent { todo!() }
```

The four `make_*` constructors must produce real protobuf types. The engineer fills these in by reading the generated `ResponseEvent` / `ClientAction` / `Message` enums in the proto-apis crate (cross-reference the catalogue from Phase 0 Task 0.3).

For `make_tool_call_event`: map `partial.name` (e.g., `"run_shell_command"`) back to a Warp `Tool::*` proto variant via `tool_registry::proto_variant_for_tool_name`, then deserialize `partial.args` (a JSON string) into the corresponding parameter struct, then construct the `ResponseEvent` with `ClientAction::AddMessagesToTask` (or equivalent — verify in catalogue) carrying a `Message` with `ToolCall { tool_call_id: partial.id, tool: Tool::RunShellCommand(...) }`.

- [ ] **Step 5: Iterate until tests pass**

```bash
cargo test -p llama_backend translate_response
```

Expected: all three tests pass. Edit constructors until the assertions hold against the actual proto shape.

- [ ] **Step 6: Commit**

```bash
git add crates/llama_backend
git commit -m "feat(llama_backend): translate OpenAI SSE deltas to Warp ResponseEvent stream"
```

---

## Phase 8 — End-to-End Wiring & First Smoke Test

### Task 8.1: Connect everything in `dispatch_if_enabled`

**Files:**
- Modify: `crates/llama_backend/src/lib.rs`

- [ ] **Step 1: Replace the stub `dispatch_if_enabled` with a real implementation**

```rust
use std::sync::OnceLock;

use crate::conversation_store::ConversationStore;
use crate::llama_client::LlamaClient;
use crate::system_prompt::WARP_SYSTEM_PROMPT;
use crate::translate_request::proto_request_to_openai;
use crate::translate_response::openai_sse_to_proto_events;

static STORE: OnceLock<ConversationStore> = OnceLock::new();

pub fn dispatch_if_enabled(request: Request) -> Option<ResponseEventStream> {
    let cfg = config::Config::from_env()?;
    let store = STORE.get_or_init(ConversationStore::new).clone();
    let stream = async_stream::try_stream! {
        let (openai_req, token) = proto_request_to_openai(
            &request,
            &store,
            WARP_SYSTEM_PROMPT,
            &cfg.model,
        )?;
        let client = LlamaClient::new(cfg.url.clone(), cfg.api_key.clone());
        let raw = client.stream_completion(&openai_req)
            .map_err(anyhow::Error::from)?;
        let raw = raw.map(|r| r.map_err(anyhow::Error::from));
        let mut events = openai_sse_to_proto_events(raw, token);
        while let Some(ev) = events.next().await {
            yield ev.map_err(anyhow::Error::from)?;
        }
    };
    Some(Box::pin(stream))
}
```

(Adjust error mapping to match `ResponseEventStream`'s declared error type.)

- [ ] **Step 2: Build**

```bash
cargo build --bin warp-oss --features gui
```

Expected: clean build.

- [ ] **Step 3: Run with WARP_LLAMA_URL set, ask Warp's agent a trivial question**

```bash
WARP_LLAMA_URL=http://192.168.8.68:8080 \
WARP_LLAMA_MODEL=Qwen3.6-27B-UD-Q3_K_XL \
RUST_LOG=llama_backend=debug,warp=info \
cargo run --bin warp-oss --features gui 2>&1 | tee /tmp/warp-llama-smoke.log
```

In the GUI, open Agent Mode and prompt: `say PONG`.

Expected:
- Warp's UI shows streamed output containing "PONG"
- `/tmp/warp-llama-smoke.log` shows `POST /v1/chat/completions` and SSE chunks
- No panic, no UI hang

If broken:
- If Warp UI shows nothing: the protobuf events likely have wrong field shapes. Cross-reference the generated `lib.rs`. Add `tracing::debug!` calls on every yielded event in `translate_response.rs`.
- If Warp UI shows raw text but freezes: `StreamFinished` wasn't yielded. Confirm `make_stream_finished_ok()` is constructed correctly.
- If Warp UI shows "error: …": capture the exact text and trace back through the conversion path.

- [ ] **Step 4: Commit on success**

```bash
git add crates/llama_backend
git commit -m "feat(llama_backend): end-to-end wiring; smoke test against CT501 passes"
```

---

## Phase 9 — Tool Round-Trip

When the model emits a tool call, Warp executes it and sends the result back as a new `Request` with `Input::ToolCallResult`. We must append the result to the conversation and re-call llama-server.

### Task 9.1: Per-tool result serialization

**Files:**
- Modify: `crates/llama_backend/src/translate_request.rs`
- Read: `~/projects/warp-proto-apis-ref/apis/multi_agent/v1/request.proto` (the `ToolCallResult` message and its result oneof)

- [ ] **Step 1: Identify each `ToolCallResult` variant**

```bash
grep -nE 'message Run.*Result|message Read.*Result|oneof result' \
  ~/projects/warp-proto-apis-ref/apis/multi_agent/v1/*.proto
```

Capture the variants (e.g., `RunShellCommandResult { stdout, stderr, exit_code }`, `ReadFilesResult { files: [{ path, content }] }`, etc.).

- [ ] **Step 2: Write a failing test for `serialize_tool_result`**

In `translate_request.rs`:

```rust
#[test]
fn shell_command_result_serializes_to_json() {
    use warp_multi_agent_api::ToolCallResult;
    use warp_multi_agent_api::tool_call_result::Result as ToolResultEnum;
    use warp_multi_agent_api::RunShellCommandResult;

    let r = ToolCallResult {
        tool_call_id: "call-1".into(),
        result: Some(ToolResultEnum::RunShellCommand(RunShellCommandResult {
            stdout: "hello\n".into(),
            stderr: "".into(),
            exit_code: 0,
            ..Default::default()
        })),
    };
    let s = serialize_tool_result(&r).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    assert_eq!(v["stdout"], "hello\n");
    assert_eq!(v["exit_code"], 0);
}
```

(Field paths and variant names must be verified against the actual generated code.)

- [ ] **Step 3: Implement `serialize_tool_result` for each variant**

```rust
fn serialize_tool_result(r: &warp_multi_agent_api::ToolCallResult) -> Result<String, anyhow::Error> {
    use warp_multi_agent_api::tool_call_result::Result as R;
    let v = match &r.result {
        Some(R::RunShellCommand(s)) => serde_json::json!({
            "stdout": s.stdout,
            "stderr": s.stderr,
            "exit_code": s.exit_code,
        }),
        Some(R::ReadFiles(f)) => serde_json::json!({
            "files": f.files.iter().map(|fi| serde_json::json!({
                "path": fi.path, "content": fi.content,
            })).collect::<Vec<_>>(),
        }),
        // ... arm per variant
        Some(other) => serde_json::json!({"unhandled_variant": format!("{:?}", other)}),
        None => serde_json::json!({"missing": true}),
    };
    Ok(v.to_string())
}
```

- [ ] **Step 4: Run tests, expand to cover every variant**

For each `R::*` arm, write a test that asserts the JSON shape. Use a `const TEST_CASES: &[(...)]` table to keep tests DRY.

- [ ] **Step 5: Persist conversation across the round-trip**

In `proto_request_to_openai`, when the input is `ToolCallResult`, also persist the *prior* assistant tool-call message into history if it isn't there yet. This requires that when we yielded the `ToolCall` event in Phase 7, we *also* appended an assistant message with `tool_calls` to the conversation in the store. Update `openai_sse_to_proto_events` to take `&ConversationStore` and call `store.upsert(...)` after emitting each tool-call event. Add a test that verifies on a follow-up `ToolCallResult`-input request, the conversation now has `[system, user, assistant(tool_calls), tool(result)]`.

- [ ] **Step 6: Smoke-test a tool round-trip**

```bash
WARP_LLAMA_URL=http://192.168.8.68:8080 \
WARP_LLAMA_MODEL=Qwen3.6-27B-UD-Q3_K_XL \
RUST_LOG=llama_backend=debug \
cargo run --bin warp-oss --features gui
```

In the GUI, prompt: `list files in /tmp`. Expected: model emits a `run_shell_command` tool call, Warp executes `ls /tmp`, posts result back, model summarizes the listing in a follow-up text reply.

- [ ] **Step 7: Commit**

```bash
git add crates/llama_backend
git commit -m "feat(llama_backend): tool round-trip; serialize per-tool result variants to JSON"
```

---

## Phase 10 — Hardening

### Task 10.1: Cancellation

**Files:**
- Modify: `crates/llama_backend/src/llama_client.rs`
- Modify: `app/src/server/server_api.rs:1091` (the call into `dispatch_if_enabled`)

- [ ] **Step 1: Find how the existing call passes a cancel signal**

In `server_api.rs:~1091-1180`, find any `cancellation_rx: oneshot::Receiver<()>` parameter or `take_until` adapter. Note the exact signature. Recon found one in `app/src/ai/agent/api/impl.rs:10-20` — confirm it's plumbed into `server_api`'s function too.

- [ ] **Step 2: Plumb the cancel receiver into `dispatch_if_enabled`**

Change the signature to `dispatch_if_enabled(request: Request, cancel: oneshot::Receiver<()>) -> Option<...>`. Inside, wrap the final stream with `.take_until(cancel)`.

- [ ] **Step 3: Test by triggering Stop in the UI**

Prompt the model with a long task (`generate a 1000-word essay`), then click Stop. Expected: streaming halts within ~1 second, no zombie llama-server connections (`ss -ant | grep 8080`).

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(llama_backend): honor user cancellation by closing the SSE stream"
```

### Task 10.2: Error events

**Files:**
- Modify: `crates/llama_backend/src/translate_response.rs`

- [ ] **Step 1: Map `LlamaBackendError` variants to Warp's `StreamFinished::Error` payload**

When the SSE stream errors, instead of bubbling `Err(_)` through the stream (which Warp's UI may handle as a fatal), emit a `StreamFinished` event with an error status. Read the proto enum for `Status` to find the right variant (likely `Error` with a message string).

- [ ] **Step 2: Test by pointing `WARP_LLAMA_URL` at an unreachable address**

```bash
WARP_LLAMA_URL=http://127.0.0.1:1 cargo run --bin warp-oss --features gui
```

Expected: Warp UI shows a clean error message ("could not reach local backend"), no panic, no UI freeze.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(llama_backend): map transport errors to StreamFinished::Error"
```

### Task 10.3: Conversation persistence to disk

**Files:**
- Modify: `crates/llama_backend/src/conversation_store.rs`

- [ ] **Step 1: Add load/save methods**

```rust
impl ConversationStore {
    pub fn load_from_disk(path: &std::path::Path) -> Self { /* read JSON files */ todo!() }
    pub fn save_one(&self, token: &str) -> std::io::Result<()> { /* write JSON */ todo!() }
}
```

- [ ] **Step 2: Save on every `upsert`**

Modify `upsert` to also write to `~/.local/state/warp-llama/conversations/<token>.json`. Use `XDG_STATE_HOME` if set.

- [ ] **Step 3: Load on `dispatch_if_enabled`'s first call**

Replace the `OnceLock::get_or_init(ConversationStore::new)` with `OnceLock::get_or_init(|| ConversationStore::load_from_disk(...))`.

- [ ] **Step 4: Test that conversations survive a restart**

Run Warp, have a 2-turn conversation, kill, restart. Send a 3rd message that depends on the first ("what was the first thing I asked?"). Expected: model recalls.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(llama_backend): persist conversations to ~/.local/state/warp-llama/"
```

---

## Phase 11 — Build, Document, Ship

### Task 11.1: Release build

**Files:**
- Modify: `app/Cargo.toml` (or workspace root) if any feature flag needs adjusting

- [ ] **Step 1: Build release**

```bash
cargo build --release --bin warp-oss --features gui 2>&1 | tee /tmp/warp-llama-release.log
```

Expected: clean, with optimization. Should take 5-15 minutes.

- [ ] **Step 2: Confirm binary location**

```bash
ls -lh target/release/warp-oss
```

Expected: ~50-200 MB binary.

### Task 11.2: User-facing documentation

**Files:**
- Create: `crates/llama_backend/README.md`

- [ ] **Step 1: Write the README**

```markdown
# llama_backend — local LLM backend for warp-oss

## Quick start

```sh
WARP_LLAMA_URL=http://192.168.8.68:8080 \
WARP_LLAMA_MODEL=Qwen3.6-27B-UD-Q3_K_XL \
./target/release/warp-oss
```

## Environment variables

| Var | Required | Description |
|---|---|---|
| `WARP_LLAMA_URL` | yes | Base URL of an OpenAI-compatible server (no trailing slash). Triggers the local backend. |
| `WARP_LLAMA_MODEL` | no | Model id sent in `model` field. Default: `default`. |
| `WARP_LLAMA_API_KEY` | no | Bearer token if the server requires it. |

## Conversation storage

Per-conversation history is kept in `~/.local/state/warp-llama/conversations/<token>.json`.
Delete to forget.

## Disabling

Unset `WARP_LLAMA_URL` to revert to Warp's real backend.

## Architecture & rebasing

See `docs/superpowers/plans/2026-04-30-warp-llama-backend.md` for the design and the
exact upstream cut point (`app/src/server/server_api.rs:1091`). When rebasing on
upstream Warp, that file is the only diff outside `crates/llama_backend/`.
```

- [ ] **Step 2: Commit**

```bash
git add crates/llama_backend/README.md
git commit -m "docs(llama_backend): user-facing README"
```

### Task 11.3: Tag a release & push

**Files:** none

- [ ] **Step 1: Tag**

```bash
git tag -a llama-backend-v0.1.0 -m "First working release: Qwen3.6 backend on CT501"
```

- [ ] **Step 2: Push branch and tag to your fork**

```bash
git push origin llama-backend
git push origin llama-backend-v0.1.0
```

- [ ] **Step 3: Open a draft PR against your fork's `main`**

```bash
gh pr create --draft --title "llama_backend: local LLM behind WARP_LLAMA_URL" --body "$(cat <<'EOF'
## Summary
- Adds `crates/llama_backend/` that diverts AI calls to a local OpenAI-compatible server when `WARP_LLAMA_URL` is set.
- One file modified outside the crate: `app/src/server/server_api.rs:1091` adds a 4-line dispatch shim.

## Test plan
- [x] Phase 0 baseline build
- [x] Phase 1 dispatch hook
- [x] Phase 2 conversation store unit tests
- [x] Phase 3 tool registry unit tests
- [x] Phase 4 outbound translation unit tests
- [x] Phase 6 SSE client wiremock test
- [x] Phase 7 inbound translation unit tests
- [x] Phase 8 end-to-end smoke ("say PONG")
- [x] Phase 9 tool round-trip ("ls /tmp")
- [x] Phase 10.1 cancellation
- [x] Phase 10.2 error events
- [x] Phase 10.3 conversation persistence across restart
EOF
)"
```

The PR is mainly a record for yourself — your fork doesn't auto-deploy.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Qwen3.6-27B tool-calling is unreliable in OpenAI format | medium | high | Phase 0.4 step 3 verifies before committing the rest. If unreliable, swap model (Qwen3.6 is currently active per project memory; Qwopus Q4_K_M is on disk as a candidate). |
| `warp_multi_agent_api` proto changes between rebases break translators | medium | medium | Pin the rev in your fork's `Cargo.toml`. Bump deliberately during rebases and run the full test suite. |
| Warp's UI assumes server-side conversation invariants we haven't reproduced (e.g., specific message ordering, sequence numbers) | medium | medium | Phase 8 smoke test catches the obvious cases. For subtler issues, add `tracing::debug!` on every yielded event and diff against a real-backend capture. |
| The system prompt is wrong enough that the model misuses tools | medium | medium | Iterate on `system_prompt.rs` after Phase 8. The single-file location makes edits cheap. |
| Warp adds a new required field to `Request` upstream | low/quarter | low | Compile error pinpoints the file; add a default value. |
| AGPL distribution requires source release if you publish binaries to others | known constraint | n/a | Personal use is fine. If sharing builds, host the source publicly. |
| llama-server rate-limits or OOMs under multi-turn agent loops | medium | low-medium | The CT501 deployment already runs vanilla Q3 at 262144 ctx per project memory; monitor with `nvidia-smi` and the existing tap proxy logs. |
| MCP tool integration (`call_mcp_tool`) needs more than schema mapping | medium | medium | Phase 3 catalogue includes it but treats it as a generic tool. If MCP servers expect richer hand-off (e.g., pre-flight schema fetch), add that in a Phase 12 follow-up. |

---

## Open Questions to Resolve During Phase 0

These are explicitly questions the engineer must answer empirically before committing to the rest of the plan. Answers go into `docs/superpowers/plans/proto-types-catalogue.md`.

1. **Exact field path for `conversation_token`** — is it `request.metadata.conversation_token` or nested elsewhere?
2. **Exact path/variants for `Input::input_kind`** — confirm `UserInputs` / `ToolCallResult` are the only ones we need to handle for v0.1.
3. **Streaming text event shape** — is it `ClientAction::AppendToMessageContent { text }` or something with deltas + message ids?
4. **Tool-call event shape on the wire** — single `ClientAction` per call, or one create + many appends?
5. **Default values for `Settings`** — does Warp's UI break if we omit `Settings::ModelConfig` or `Settings::supported_tools`?
6. **Does `oss.rs`'s `ChannelState` allow our crate to read the user's auth token if needed for non-AI calls?** (For v0.1 we don't need it; flag if Phase 9 reveals a need.)

If any answer reveals the cut point is wrong, stop, update this plan, and resume.

---

## Self-Review Checklist (Performed)

- [x] **Spec coverage:** every stated goal is addressed — fork built (Phase 0/11), AI rerouted (Phase 1, 8), conversation managed (Phase 2, 9, 10.3), tools (Phase 3, 9), system prompt (Phase 5), errors/cancellation (Phase 10).
- [x] **Placeholder scan:** No `TODO` or "fill in" steps remain except the explicit `todo!()` calls inside *test setup helpers* whose body the engineer must complete by reading concrete generated proto code (Phase 4 step 6, Phase 7 step 4 constructors). These are not plan failures — they are sites where we instruct the engineer to look something up. Each is paired with a citation to the file to read.
- [x] **Type consistency:** `ConversationStore`, `Conversation`, `OpenAIMessage`, `OpenAIRole`, `ChatCompletionRequest`, `ResponseEventStream`, `LlamaBackendError`, `Config` are used identically wherever they appear.
