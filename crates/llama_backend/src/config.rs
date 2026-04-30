use std::env;

/// Runtime configuration for the local-backend dispatch, populated from
/// process env vars at the start of each turn.
///
/// `WARP_LLAMA_URL` is the trigger: if unset, [`Config::from_env`] returns
/// `None` and the dispatch falls through to Warp's real backend.
#[derive(Debug, Clone)]
pub struct Config {
    /// Base URL of the OpenAI-compatible server, no trailing slash
    /// (e.g. `http://192.168.8.68:8080`).
    pub url: String,
    /// Model id passed in the `model` field of `/v1/chat/completions`.
    pub model: String,
    /// Optional bearer token. Set via `WARP_LLAMA_API_KEY`.
    pub api_key: Option<String>,
}

impl Config {
    pub fn from_env() -> Option<Self> {
        let url = env::var("WARP_LLAMA_URL").ok()?;
        let url = url.trim_end_matches('/').to_string();
        let model =
            env::var("WARP_LLAMA_MODEL").unwrap_or_else(|_| "default".to_string());
        let api_key = env::var("WARP_LLAMA_API_KEY").ok().filter(|s| !s.is_empty());
        Some(Self {
            url,
            model,
            api_key,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Env tests need to be serialized: env-var mutations are process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env<F: FnOnce() -> R, R>(
        vars: &[(&str, Option<&str>)],
        f: F,
    ) -> R {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved: Vec<_> = vars
            .iter()
            .map(|(k, _)| (k.to_string(), env::var(k).ok()))
            .collect();
        for (k, v) in vars {
            match v {
                Some(val) => env::set_var(k, val),
                None => env::remove_var(k),
            }
        }
        let out = f();
        for (k, v) in saved {
            match v {
                Some(val) => env::set_var(&k, val),
                None => env::remove_var(&k),
            }
        }
        out
    }

    #[test]
    fn from_env_returns_none_when_url_unset() {
        with_env(&[("WARP_LLAMA_URL", None)], || {
            assert!(Config::from_env().is_none());
        });
    }

    #[test]
    fn from_env_strips_trailing_slash() {
        with_env(
            &[
                ("WARP_LLAMA_URL", Some("http://example.com:8080/")),
                ("WARP_LLAMA_MODEL", None),
                ("WARP_LLAMA_API_KEY", None),
            ],
            || {
                let cfg = Config::from_env().expect("WARP_LLAMA_URL set");
                assert_eq!(cfg.url, "http://example.com:8080");
                assert_eq!(cfg.model, "default");
                assert!(cfg.api_key.is_none());
            },
        );
    }

    #[test]
    fn from_env_picks_up_model_and_key() {
        with_env(
            &[
                ("WARP_LLAMA_URL", Some("http://h:1")),
                ("WARP_LLAMA_MODEL", Some("Q3K")),
                ("WARP_LLAMA_API_KEY", Some("secret")),
            ],
            || {
                let cfg = Config::from_env().unwrap();
                assert_eq!(cfg.model, "Q3K");
                assert_eq!(cfg.api_key.as_deref(), Some("secret"));
            },
        );
    }

    #[test]
    fn empty_api_key_is_treated_as_unset() {
        with_env(
            &[
                ("WARP_LLAMA_URL", Some("http://h")),
                ("WARP_LLAMA_API_KEY", Some("")),
            ],
            || {
                let cfg = Config::from_env().unwrap();
                assert!(cfg.api_key.is_none());
            },
        );
    }
}
