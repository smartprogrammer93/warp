# `llama_backend` — local-LLM backend for `warp-oss`

A drop-in backend for the open-source Warp terminal that diverts Agent-Mode
traffic from `https://app.warp.dev/ai/multi-agent` to a self-hosted
OpenAI-compatible server (e.g. [`llama.cpp`'s
`llama-server`](https://github.com/ggml-org/llama.cpp), vLLM, Ollama via its
OpenAI compat layer).

The diversion is **gated by env var**: with `WARP_LLAMA_URL` unset, the
binary behaves exactly like upstream `warp-oss`. With it set, AI turns are
served by your local model — no Warp account required for the AI features
(other Warp features like Drive/sync still hit `app.warp.dev` and need an
account).

## Quick start

```sh
cargo build --release --bin warp-oss --features gui

WARP_LLAMA_URL=http://192.168.8.68:8080 \
WARP_LLAMA_MODEL=Qwen3.6-27B-UD-Q4_K_XL.gguf \
WARP_LLAMA_API_KEY=your-bearer-token \
./target/release/warp-oss
```

Open Agent Mode and prompt as usual. The first turn mints a fresh
conversation id and sends `[system, user]` to the server; subsequent turns
preserve the in-process history (and persist it to disk — see below).

## Environment variables

| Var                  | Required | Default     | Notes |
|----------------------|---------:|-------------|-------|
| `WARP_LLAMA_URL`     | ✅        | —           | Base URL, no trailing slash. Setting this engages the local backend; unsetting reverts to `app.warp.dev`. |
| `WARP_LLAMA_MODEL`   | —        | `default`   | Sent in the `model` field of `/v1/chat/completions`. |
| `WARP_LLAMA_API_KEY` | —        | unset       | Bearer token if the server requires it. Pass `WARP_LLAMA_API_KEY=""` to force unauthenticated. |
| `RUST_LOG`           | —        | unset       | `RUST_LOG=llama_backend=debug` enables crate-level traces. |

## Conversations

Per-conversation history is persisted to:

```
$XDG_STATE_HOME/warp-llama/conversations/<conversation_id>.json
```

(falling back to `~/.local/state/warp-llama/conversations/` if `XDG_STATE_HOME`
is unset). Files are written atomically (tempfile + rename) on every
assistant turn. Delete a file to forget that conversation; delete the
directory to forget all of them.

## Tools

Six built-in tools are exposed to the model in v0.1:

- `run_shell_command`
- `read_files`
- `search_codebase`
- `apply_file_diffs` (search/replace + new files + deleted files; the v4a hunk format is not exposed)
- `grep`
- `file_glob_v2`

…intersected with `Settings.supported_tools` from the request (Warp's UI
declares what it can execute). MCP tools are merged in dynamically from
`MCPContext.servers[].tools[]`; their names are encoded as
`mcp__<server_id>__<tool_name>` so the response translator can route the
model's pick to `Tool::CallMcpTool { name, args, server_id }`.

The other 25 `ToolType` variants Warp defines (suggest_plan, use_computer,
start_agent, ask_user_question, …) are intentionally out of scope for v0.1 —
the model never sees them.

## Architecture

```
warp-oss (UI)
  └─ ServerApi::generate_multi_agent_output           # app/src/server/server_api.rs:1091
       └─ llama_backend::dispatch_if_enabled          # 4-line shim, only if WARP_LLAMA_URL
            ├─ translate_request                       # proto Request -> OpenAI ChatCompletionRequest
            ├─ ConversationStore (in-memory + JSON)   # turn-by-turn history rebuild
            ├─ tool_registry                           # static tools ⨯ supported_tools ⨯ MCP merge
            ├─ LlamaClient (reqwest + reqwest_eventsource)
            └─ translate_response                      # OpenAI SSE -> StreamInit / Begin /
                                                       # CreateTask / AddMessages /
                                                       # AppendToMessageContent×N / tool calls /
                                                       # Commit / StreamFinished
```

The `dispatch_if_enabled` returning `Option<DispatchStream>` is the entire
integration surface. Outside `app/src/server/server_api.rs:1091` only one
line of upstream code is touched (the `app/Cargo.toml` dep). The crate is
self-contained at `crates/llama_backend/`.

## Running tests

```sh
cargo test -p llama_backend                          # 54 unit tests
cargo test -p llama_backend --test live_smoke -- --ignored --nocapture --test-threads=1
```

The live smoke tests (`say_pong_against_real_llama_server`,
`tool_round_trip_list_tmp_against_real_llama_server`) require
`WARP_LLAMA_URL`, `WARP_LLAMA_MODEL`, and `WARP_LLAMA_API_KEY` to be
exported.

## Rebasing on upstream Warp

When upstream bumps versions:

1. `git fetch upstream main && git rebase upstream/main`
2. The conflict surface is one file: `app/src/server/server_api.rs` near
   line 1091 (the dispatch shim) and `app/Cargo.toml` (the `llama_backend =
   { path = ... }` dep).
3. `Cargo.toml` (workspace) pins `warp_multi_agent_api` to a specific rev.
   If upstream bumps it, our translators may need updating — the proto
   schema lives in `github.com/warpdotdev/warp-proto-apis`. Compile errors
   pinpoint any field renames.
4. `cargo test -p llama_backend` and the live smoke before merging.

## License

Per AGPL-3.0 (inherited from upstream Warp). If you redistribute binaries
with the local backend enabled, you must publish the source.
