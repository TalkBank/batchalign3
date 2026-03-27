#![warn(missing_docs)]
//! Binary entry point for the `batchalign3` command-line tool.
//!
//! This crate is the thin executable wrapper that ties together the library
//! crates in the root Rust workspace. It is responsible for:
//!
//! 1. **Argument parsing** -- delegates to `batchalign_cli::args::Cli` (clap).
//! 2. **Tracing initialization** -- configures a `tracing_subscriber` filter
//!    based on the `-v` verbosity flag (`warn` / `info` / `debug` / `trace`).
//!    When `BATCHALIGN_OTLP_ENABLE` or an OTLP endpoint env var is set, an
//!    OpenTelemetry span exporter is layered on top for distributed tracing.
//! 3. **Subcommand dispatch** -- delegates to [`batchalign_cli::run_command`],
//!    the single source of truth shared with the PyO3 console_scripts entry point.
//! 4. **Exit code propagation** -- maps `CliError` variants to appropriate
//!    numeric exit codes so callers (scripts, CI) can distinguish failure modes.
//!
//! The binary itself never loads ML models or processes CHAT files directly.
//! Processing commands route either into direct in-process execution via
//! `DirectHost` or into an explicit server, both of which delegate ML work to
//! Python worker processes.

use clap::Parser;
use opentelemetry::KeyValue;
use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

use batchalign_cli::args::Cli;

#[derive(Debug, Clone, PartialEq, Eq)]
struct OtlpRuntimeConfig {
    enabled: bool,
    batchalign_endpoint: Option<String>,
    otel_exporter_endpoint: Option<String>,
}

impl OtlpRuntimeConfig {
    fn from_env() -> Self {
        Self::from_sources(
            std::env::var("BATCHALIGN_OTLP_ENABLE").ok().as_deref(),
            std::env::var("BATCHALIGN_OTLP_ENDPOINT").ok().as_deref(),
            std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok().as_deref(),
        )
    }

    fn from_sources(
        enabled: Option<&str>,
        batchalign_endpoint: Option<&str>,
        otel_exporter_endpoint: Option<&str>,
    ) -> Self {
        Self {
            enabled: parse_env_bool(enabled),
            batchalign_endpoint: normalized_env_value(batchalign_endpoint),
            otel_exporter_endpoint: normalized_env_value(otel_exporter_endpoint),
        }
    }

    fn should_enable(&self) -> bool {
        self.enabled || self.batchalign_endpoint.is_some() || self.otel_exporter_endpoint.is_some()
    }

    fn batchalign_endpoint(&self) -> Option<&str> {
        self.batchalign_endpoint.as_deref()
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let tracer_provider = init_tracing(cli.global.verbose);

    let update_handle = batchalign_cli::update_check::spawn_update_check();

    let exit_code = match batchalign_cli::run_command(cli).await {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            e.exit_code()
        }
    };

    // Give the background update check a moment to finish printing.
    let _ = tokio::time::timeout(std::time::Duration::from_millis(500), update_handle).await;

    if let Some(provider) = tracer_provider
        && let Err(err) = provider.shutdown()
    {
        eprintln!("warning: telemetry shutdown failed: {err}");
    }

    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

fn init_tracing(verbose: u8) -> Option<SdkTracerProvider> {
    let filter = match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter));

    let otlp = OtlpRuntimeConfig::from_env();
    if otlp.should_enable() {
        match init_otlp_provider(&otlp) {
            Ok(provider) => {
                let tracer = provider.tracer("batchalign3");
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(tracing_subscriber::fmt::layer().with_target(false))
                    .with(tracing_opentelemetry::layer().with_tracer(tracer))
                    .init();
                return Some(provider);
            }
            Err(message) => {
                eprintln!("warning: OTLP tracing disabled: {message}");
            }
        }
    }

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();
    None
}

fn init_otlp_provider(config: &OtlpRuntimeConfig) -> Result<SdkTracerProvider, String> {
    let mut exporter_builder = opentelemetry_otlp::SpanExporter::builder().with_http();
    if let Some(endpoint) = config.batchalign_endpoint() {
        exporter_builder = exporter_builder.with_endpoint(endpoint.to_string());
    }

    let exporter = exporter_builder
        .build()
        .map_err(|err| format!("failed to build OTLP span exporter: {err}"))?;
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            Resource::builder_empty()
                .with_attributes([
                    KeyValue::new("service.name", "batchalign3"),
                    KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                ])
                .build(),
        )
        .build();

    global::set_tracer_provider(provider.clone());
    Ok(provider)
}

fn normalized_env_value(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn parse_env_bool(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::OtlpRuntimeConfig;

    #[test]
    fn otlp_runtime_config_enables_from_flag() {
        let config = OtlpRuntimeConfig::from_sources(Some("yes"), None, None);
        assert!(config.should_enable());
        assert!(config.batchalign_endpoint().is_none());
    }

    #[test]
    fn otlp_runtime_config_enables_from_either_endpoint() {
        let config =
            OtlpRuntimeConfig::from_sources(None, Some("http://collector"), Some("http://otel"));
        assert!(config.should_enable());
        assert_eq!(config.batchalign_endpoint(), Some("http://collector"));
    }

    #[test]
    fn otlp_runtime_config_ignores_blank_values() {
        let config = OtlpRuntimeConfig::from_sources(Some(" "), Some(" "), Some(""));
        assert!(!config.should_enable());
    }
}
