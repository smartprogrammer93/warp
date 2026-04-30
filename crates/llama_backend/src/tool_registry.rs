//! The static set of Warp tools we expose to the local model, plus the
//! bidirectional mapping between OpenAI function-call names and Warp's
//! `Tool` proto-variant names.
//!
//! v0.1 supports six built-in tools (`run_shell_command`, `read_files`,
//! `search_codebase`, `apply_file_diffs`, `grep`, `file_glob_v2`) plus
//! dynamically-merged MCP tools, named `mcp__<server_id>__<tool_name>` so
//! [`parse_mcp_tool_name`] can route them back to `Tool::CallMcpTool`.
//!
//! `tools_for_request` performs the per-request intersection: our static set
//! is filtered by `Settings.supported_tools`, then merged with
//! `MCPContext.servers[].tools[]` from the request.

use crate::openai_types::{OpenAIFunctionDef, OpenAIToolDef};
use serde_json::json;
use warp_multi_agent_api::{Request, ToolType};

/// Single source of truth for our static tool set: each entry is
/// `(openai_tool_name, proto_variant_name, ToolType, json_schema_factory)`.
///
/// `proto_variant_name` is the PascalCase name of the variant inside
/// `warp_multi_agent_api::message::tool_call::Tool` (e.g. `RunShellCommand`).
/// We use string names rather than the typed variant because the response
/// translator doesn't know which variant it'll need until it sees the model's
/// reply, and matching on the string is more compact than a type-level switch.
type SchemaFn = fn() -> serde_json::Value;
struct ToolSpec {
    openai_name: &'static str,
    proto_variant: &'static str,
    proto_tool_type: ToolType,
    description: &'static str,
    schema: SchemaFn,
}

const TOOL_TABLE: &[ToolSpec] = &[
    ToolSpec {
        openai_name: "run_shell_command",
        proto_variant: "RunShellCommand",
        proto_tool_type: ToolType::RunShellCommand,
        description: "Execute a shell command on the user's machine and return stdout, stderr, and exit code. Prefer this for anything you'd type at a prompt; do not run interactive editors or pagers.",
        schema: schema_run_shell_command,
    },
    ToolSpec {
        openai_name: "read_files",
        proto_variant: "ReadFiles",
        proto_tool_type: ToolType::ReadFiles,
        description: "Read one or more files into context. Use this before editing.",
        schema: schema_read_files,
    },
    ToolSpec {
        openai_name: "search_codebase",
        proto_variant: "SearchCodebase",
        proto_tool_type: ToolType::SearchCodebase,
        description: "Semantic search over the codebase. Use for natural-language queries when you don't know the exact symbol.",
        schema: schema_search_codebase,
    },
    ToolSpec {
        openai_name: "apply_file_diffs",
        proto_variant: "ApplyFileDiffs",
        proto_tool_type: ToolType::ApplyFileDiffs,
        description: "Edit files by applying search/replace diffs, creating new files, or deleting files. Always read affected files first.",
        schema: schema_apply_file_diffs,
    },
    ToolSpec {
        openai_name: "grep",
        proto_variant: "Grep",
        proto_tool_type: ToolType::Grep,
        description: "Literal/regex pattern search across files in the project.",
        schema: schema_grep,
    },
    ToolSpec {
        openai_name: "file_glob_v2",
        proto_variant: "FileGlobV2",
        proto_tool_type: ToolType::FileGlobV2,
        description: "List files whose paths match glob patterns (supports ?, *, []).",
        schema: schema_file_glob_v2,
    },
];

/// Prefix marking an OpenAI tool name as referring to an MCP server tool.
const MCP_PREFIX: &str = "mcp__";
/// Separator between server_id and tool name in MCP-encoded names.
const MCP_SEP: &str = "__";

/// Build the full OpenAI tools list for a given request.
///
/// Static tools are intersected with `Settings.supported_tools` (an empty
/// list means the proto contract "no restriction" — we include all of ours),
/// and MCP tools are merged dynamically with name-encoding so the response
/// translator can route them.
pub fn tools_for_request(request: &Request) -> Vec<OpenAIToolDef> {
    let mut out = Vec::new();

    // 1. Static tools, filtered by Settings.supported_tools.
    let supported: Option<Vec<ToolType>> = request.settings.as_ref().and_then(|s| {
        if s.supported_tools.is_empty() {
            None
        } else {
            Some(
                s.supported_tools
                    .iter()
                    .filter_map(|i| ToolType::try_from(*i).ok())
                    .collect(),
            )
        }
    });

    for spec in TOOL_TABLE {
        let allowed = match &supported {
            None => true,
            Some(set) => set.iter().any(|t| *t == spec.proto_tool_type),
        };
        if allowed {
            out.push(static_tool_def(spec));
        }
    }

    // 2. MCP tools from MCPContext.servers[].tools[].
    if let Some(mcp) = request.mcp_context.as_ref() {
        for server in &mcp.servers {
            for tool in &server.tools {
                let encoded = make_mcp_tool_name(&server.id, &tool.name);
                let parameters = tool
                    .input_schema
                    .as_ref()
                    .map(prost_struct_to_json)
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
                out.push(OpenAIToolDef {
                    kind: "function".to_string(),
                    function: OpenAIFunctionDef {
                        name: encoded,
                        description: if tool.description.is_empty() {
                            format!("MCP tool '{}' on server '{}'", tool.name, server.name)
                        } else {
                            tool.description.clone()
                        },
                        parameters,
                    },
                });
            }
        }
    }

    out
}

fn static_tool_def(spec: &ToolSpec) -> OpenAIToolDef {
    OpenAIToolDef {
        kind: "function".to_string(),
        function: OpenAIFunctionDef {
            name: spec.openai_name.to_string(),
            description: spec.description.to_string(),
            parameters: (spec.schema)(),
        },
    }
}

/// Look up the proto-variant name for an OpenAI tool name picked by the
/// model. Returns `Some("CallMcpTool")` for any name with the MCP prefix,
/// `Some("<Variant>")` for static tools, `None` for unknown.
pub fn proto_variant_for_tool_name(name: &str) -> Option<&'static str> {
    if is_mcp_tool_name(name) {
        return Some("CallMcpTool");
    }
    TOOL_TABLE
        .iter()
        .find(|s| s.openai_name == name)
        .map(|s| s.proto_variant)
}

/// Inverse of `proto_variant_for_tool_name` for static tools (used when
/// translating an emitted Warp tool-call back to an OpenAI tool name in
/// stored conversation history during a follow-up turn).
pub fn tool_name_for_proto_variant(variant: &str) -> Option<&'static str> {
    TOOL_TABLE
        .iter()
        .find(|s| s.proto_variant == variant)
        .map(|s| s.openai_name)
}

/// Encode `(server_id, tool_name)` into a single OpenAI-tools-list-safe name.
pub fn make_mcp_tool_name(server_id: &str, tool_name: &str) -> String {
    format!("{MCP_PREFIX}{server_id}{MCP_SEP}{tool_name}")
}

/// Decode an MCP-encoded OpenAI tool name back into `(server_id, tool_name)`,
/// or `None` if it isn't MCP-prefixed.
pub fn parse_mcp_tool_name(name: &str) -> Option<(&str, &str)> {
    let rest = name.strip_prefix(MCP_PREFIX)?;
    let sep = rest.find(MCP_SEP)?;
    let (server_id, after) = rest.split_at(sep);
    Some((server_id, &after[MCP_SEP.len()..]))
}

pub fn is_mcp_tool_name(name: &str) -> bool {
    name.starts_with(MCP_PREFIX) && name[MCP_PREFIX.len()..].contains(MCP_SEP)
}

/// Build a Warp `tool_call::Tool` proto variant from the model's chosen
/// OpenAI tool name and the JSON-encoded arguments string the model emitted.
///
/// Returns `None` for unknown tool names (caller should drop the tool call
/// gracefully). Returns `Err` if the args JSON is malformed.
pub fn build_proto_tool(
    name: &str,
    args_json: &str,
) -> anyhow::Result<
    Option<warp_multi_agent_api::message::tool_call::Tool>,
> {
    use warp_multi_agent_api::message::tool_call::Tool as T;
    use warp_multi_agent_api::message::tool_call::*;

    // MCP tools first: any encoded-name with `mcp__<server>__<name>` prefix.
    if let Some((server_id, real_name)) = parse_mcp_tool_name(name) {
        let args_value: serde_json::Value =
            serde_json::from_str(args_json).unwrap_or(serde_json::Value::Object(Default::default()));
        let args_struct = json_to_prost_struct(&args_value);
        return Ok(Some(T::CallMcpTool(CallMcpTool {
            name: real_name.to_string(),
            args: Some(args_struct),
            server_id: server_id.to_string(),
        })));
    }

    // Static tools: parse args as JSON and pluck the documented fields.
    let v: serde_json::Value = serde_json::from_str(args_json)
        .map_err(|e| anyhow::anyhow!("tool {name} args_json invalid: {e}; raw={args_json:?}"))?;

    let s = |k: &str| -> String {
        v.get(k)
            .and_then(|x| x.as_str())
            .map(str::to_string)
            .unwrap_or_default()
    };
    let strs = |k: &str| -> Vec<String> {
        v.get(k)
            .and_then(|x| x.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let i32f = |k: &str| -> i32 {
        v.get(k).and_then(|x| x.as_i64()).unwrap_or(0) as i32
    };

    let tool = match name {
        "run_shell_command" => T::RunShellCommand(RunShellCommand {
            command: s("command"),
            ..Default::default()
        }),
        "read_files" => {
            let files = v
                .get("files")
                .and_then(|x| x.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|f| {
                            let name = f.get("name").and_then(|x| x.as_str())?;
                            Some(read_files::File {
                                name: name.to_string(),
                                ..Default::default()
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            T::ReadFiles(ReadFiles { files })
        }
        "search_codebase" => T::SearchCodebase(SearchCodebase {
            query: s("query"),
            path_filters: strs("path_filters"),
            codebase_path: s("codebase_path"),
        }),
        "apply_file_diffs" => {
            let diffs = v
                .get("diffs")
                .and_then(|x| x.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|d| apply_file_diffs::FileDiff {
                            file_path: d
                                .get("file_path")
                                .and_then(|x| x.as_str())
                                .unwrap_or_default()
                                .to_string(),
                            search: d
                                .get("search")
                                .and_then(|x| x.as_str())
                                .unwrap_or_default()
                                .to_string(),
                            replace: d
                                .get("replace")
                                .and_then(|x| x.as_str())
                                .unwrap_or_default()
                                .to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let new_files = v
                .get("new_files")
                .and_then(|x| x.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|f| apply_file_diffs::NewFile {
                            file_path: f
                                .get("file_path")
                                .and_then(|x| x.as_str())
                                .unwrap_or_default()
                                .to_string(),
                            content: f
                                .get("content")
                                .and_then(|x| x.as_str())
                                .unwrap_or_default()
                                .to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let deleted_files = v
                .get("deleted_files")
                .and_then(|x| x.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|f| apply_file_diffs::DeleteFile {
                            file_path: f
                                .get("file_path")
                                .and_then(|x| x.as_str())
                                .unwrap_or_default()
                                .to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            T::ApplyFileDiffs(ApplyFileDiffs {
                summary: s("summary"),
                diffs,
                new_files,
                deleted_files,
                v4a_updates: vec![],
            })
        }
        "grep" => T::Grep(Grep {
            queries: strs("queries"),
            path: s("path"),
        }),
        "file_glob_v2" => T::FileGlobV2(FileGlobV2 {
            patterns: strs("patterns"),
            search_dir: s("search_dir"),
            max_matches: i32f("max_matches"),
            max_depth: i32f("max_depth"),
            min_depth: i32f("min_depth"),
        }),
        _ => return Ok(None),
    };
    Ok(Some(tool))
}

/// Convert a `serde_json::Value` to a `prost_types::Struct` (top-level must
/// be an object; non-object inputs become empty structs).
fn json_to_prost_struct(v: &serde_json::Value) -> prost_types::Struct {
    let mut fields = std::collections::BTreeMap::new();
    if let Some(obj) = v.as_object() {
        for (k, vv) in obj {
            fields.insert(k.clone(), json_to_prost_value(vv));
        }
    }
    prost_types::Struct { fields }
}

fn json_to_prost_value(v: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match v {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(b) => Kind::BoolValue(*b),
        serde_json::Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Kind::StringValue(s.clone()),
        serde_json::Value::Array(arr) => Kind::ListValue(prost_types::ListValue {
            values: arr.iter().map(json_to_prost_value).collect(),
        }),
        serde_json::Value::Object(_) => Kind::StructValue(json_to_prost_struct(v)),
    };
    prost_types::Value { kind: Some(kind) }
}

/// Convert a `prost_types::Struct` (the proto wire form for arbitrary JSON)
/// into a `serde_json::Value`. MCP servers send their input schemas as
/// google.protobuf.Struct so the LLM `parameters` schema can be passed
/// through without a typed Rust mirror.
fn prost_struct_to_json(s: &prost_types::Struct) -> serde_json::Value {
    let mut map = serde_json::Map::with_capacity(s.fields.len());
    for (k, v) in &s.fields {
        map.insert(k.clone(), prost_value_to_json(v));
    }
    serde_json::Value::Object(map)
}

fn prost_value_to_json(v: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match &v.kind {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::NumberValue(n)) => serde_json::Number::from_f64(*n)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::StructValue(s)) => prost_struct_to_json(s),
        Some(Kind::ListValue(l)) => {
            serde_json::Value::Array(l.values.iter().map(prost_value_to_json).collect())
        }
    }
}

// ---------------- per-tool schema factories ----------------

fn schema_run_shell_command() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command line to execute."
            }
        },
        "required": ["command"]
    })
}

fn schema_read_files() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Path to a file relative to the working directory."
                        }
                    },
                    "required": ["name"]
                }
            }
        },
        "required": ["files"]
    })
}

fn schema_search_codebase() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query describing what to find."
            },
            "path_filters": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional path globs to restrict the search."
            },
            "codebase_path": {
                "type": "string",
                "description": "Optional absolute path to the codebase root; defaults to cwd."
            }
        },
        "required": ["query"]
    })
}

fn schema_apply_file_diffs() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One-sentence summary of the change set."
            },
            "diffs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "search":    {"type": "string", "description": "Exact text to replace (must match verbatim)."},
                        "replace":   {"type": "string", "description": "Replacement text."}
                    },
                    "required": ["file_path", "search", "replace"]
                }
            },
            "new_files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content":   {"type": "string"}
                    },
                    "required": ["file_path", "content"]
                }
            },
            "deleted_files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"}
                    },
                    "required": ["file_path"]
                }
            }
        },
        "required": ["summary"]
    })
}

fn schema_grep() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Patterns to grep for."
            },
            "path": {
                "type": "string",
                "description": "Path (file or directory) to search within."
            }
        },
        "required": ["queries", "path"]
    })
}

fn schema_file_glob_v2() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Glob patterns (?, *, []) to match file names against."
            },
            "search_dir": {
                "type": "string",
                "description": "Directory to search within."
            },
            "max_matches": {"type": "integer", "minimum": 0},
            "max_depth":   {"type": "integer", "minimum": 0},
            "min_depth":   {"type": "integer", "minimum": 0}
        },
        "required": ["patterns", "search_dir"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_multi_agent_api::request::{self, mcp_context::{McpServer, McpTool}};
    use warp_multi_agent_api::Request;

    fn req_with_settings(supported: Vec<ToolType>) -> Request {
        let supported_i32: Vec<i32> = supported.into_iter().map(|t| t as i32).collect();
        Request {
            settings: Some(request::Settings {
                supported_tools: supported_i32,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn empty_supported_tools_includes_all_six_static() {
        let req = req_with_settings(vec![]);
        let tools = tools_for_request(&req);
        let names: Vec<_> = tools.iter().map(|t| t.function.name.as_str()).collect();
        for expect in [
            "run_shell_command",
            "read_files",
            "search_codebase",
            "apply_file_diffs",
            "grep",
            "file_glob_v2",
        ] {
            assert!(names.contains(&expect), "missing {expect} in {names:?}");
        }
        assert_eq!(tools.len(), 6);
    }

    #[test]
    fn supported_tools_filters_static_set() {
        let req = req_with_settings(vec![ToolType::RunShellCommand, ToolType::Grep]);
        let tools = tools_for_request(&req);
        let names: Vec<_> = tools.iter().map(|t| t.function.name.as_str()).collect();
        assert_eq!(
            names.iter().copied().collect::<std::collections::HashSet<_>>(),
            ["run_shell_command", "grep"].iter().copied().collect()
        );
    }

    #[test]
    fn run_shell_command_schema_requires_command() {
        let req = req_with_settings(vec![]);
        let tools = tools_for_request(&req);
        let t = tools
            .iter()
            .find(|t| t.function.name == "run_shell_command")
            .expect("run_shell_command present");
        let req_field = t.function.parameters.get("required").unwrap().as_array().unwrap();
        assert_eq!(req_field.len(), 1);
        assert_eq!(req_field[0], "command");
    }

    #[test]
    fn name_routing_roundtrip_for_static_tools() {
        for (n, v) in [
            ("run_shell_command", "RunShellCommand"),
            ("read_files", "ReadFiles"),
            ("search_codebase", "SearchCodebase"),
            ("apply_file_diffs", "ApplyFileDiffs"),
            ("grep", "Grep"),
            ("file_glob_v2", "FileGlobV2"),
        ] {
            assert_eq!(proto_variant_for_tool_name(n), Some(v), "fwd {n}");
            assert_eq!(tool_name_for_proto_variant(v), Some(n), "rev {v}");
        }
        assert_eq!(proto_variant_for_tool_name("nope"), None);
        assert_eq!(tool_name_for_proto_variant("DoesNotExist"), None);
    }

    #[test]
    fn mcp_tool_names_encode_and_decode() {
        let encoded = make_mcp_tool_name("sentry-001", "create_issue");
        assert_eq!(encoded, "mcp__sentry-001__create_issue");
        assert!(is_mcp_tool_name(&encoded));
        let (server, name) = parse_mcp_tool_name(&encoded).unwrap();
        assert_eq!(server, "sentry-001");
        assert_eq!(name, "create_issue");
        assert!(parse_mcp_tool_name("not_mcp").is_none());
        assert!(parse_mcp_tool_name("mcp__only_one").is_none());
        assert_eq!(proto_variant_for_tool_name(&encoded), Some("CallMcpTool"));
    }

    #[test]
    fn mcp_tool_with_underscores_in_tool_name_decodes_correctly() {
        let encoded = make_mcp_tool_name("github", "create_pull_request");
        let (server, name) = parse_mcp_tool_name(&encoded).unwrap();
        assert_eq!(server, "github");
        assert_eq!(name, "create_pull_request");
    }

    #[test]
    fn mcp_context_tools_are_merged_into_request_tools() {
        let mut req = req_with_settings(vec![]);
        req.mcp_context = Some(request::McpContext {
            servers: vec![McpServer {
                id: "linear".into(),
                name: "Linear".into(),
                description: String::new(),
                resources: vec![],
                tools: vec![McpTool {
                    name: "create_issue".into(),
                    description: "Create a Linear issue.".into(),
                    input_schema: None,
                }],
            }],
            ..Default::default()
        });
        let tools = tools_for_request(&req);
        let names: Vec<_> = tools.iter().map(|t| t.function.name.as_str()).collect();
        assert!(names.contains(&"mcp__linear__create_issue"), "got {names:?}");
        let mcp_tool = tools
            .iter()
            .find(|t| t.function.name == "mcp__linear__create_issue")
            .unwrap();
        assert_eq!(mcp_tool.function.description, "Create a Linear issue.");
    }
}
