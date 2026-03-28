//! HTTP client for communicating with batchalign servers.
//!
//! Wraps `reqwest::Client` with retry logic matching the Python implementation.

use std::time::Duration;

use batchalign_app::api::{
    DisplayPath, FileResult, HealthResponse, JobId, JobInfo, JobListItem, JobResultResponse,
    JobSubmission,
};
use reqwest::Client;
use tracing::debug;

use crate::error::CliError;

// ---------------------------------------------------------------------------
// Constants (matching Python `dispatch_server.py`)
// ---------------------------------------------------------------------------

/// Minimum poll interval (seconds).
pub const POLL_MIN: f64 = 0.5;
/// Maximum poll interval (seconds).
pub const POLL_MAX: f64 = 5.0;
/// Poll interval step increase per idle poll.
pub const POLL_STEP: f64 = 0.5;
/// Maximum retry attempts for transient errors.
pub const RETRY_ATTEMPTS: u32 = 3;
/// Initial retry backoff (seconds), doubles each attempt.
pub const RETRY_BACKOFF: f64 = 2.0;
/// Consecutive poll failures before giving up.
pub const MAX_POLL_FAILURES: u32 = 10;

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// HTTP client for a batchalign server.
#[derive(Clone)]
pub struct BatchalignClient {
    http: Client,
}

impl BatchalignClient {
    /// Create a new client with default timeout settings.
    pub fn new() -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build HTTP client");
        Self { http }
    }

    /// `GET /health` — check server health and capabilities.
    pub async fn health_check(&self, url: &str) -> Result<HealthResponse, CliError> {
        let resp = self
            .request_with_retry(reqwest::Method::GET, &format!("{url}/health"), None::<&()>)
            .await?;
        let health: HealthResponse = resp.json().await?;
        Ok(health)
    }

    /// `POST /jobs` — submit a new job.
    pub async fn submit_job(&self, url: &str, sub: &JobSubmission) -> Result<JobInfo, CliError> {
        let resp = self
            .http
            .post(format!("{url}/jobs"))
            .json(sub)
            .timeout(Duration::from_secs(120))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let detail = read_http_error_detail(resp).await;
            return Err(CliError::ServerHttp { status, detail });
        }

        let info: JobInfo = resp.json().await?;
        Ok(info)
    }

    /// `GET /jobs/{id}` — get job status.
    pub async fn get_job(&self, url: &str, job_id: &JobId) -> Result<JobInfo, CliError> {
        let resp = self
            .http
            .get(format!("{url}/jobs/{job_id}"))
            .timeout(Duration::from_secs(10))
            .send()
            .await?;

        if resp.status().as_u16() == 404 {
            return Err(CliError::JobLost {
                job_id: job_id.clone(),
            });
        }
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let detail = read_http_error_detail(resp).await;
            return Err(CliError::ServerHttp { status, detail });
        }

        let info: JobInfo = resp.json().await?;
        Ok(info)
    }

    /// `GET /jobs/{id}/results/{*filename}` — fetch a single file result.
    ///
    /// The filename is embedded directly in the URL path (slashes included).
    /// The server route uses axum's `{*filename}` wildcard to capture the full
    /// remaining path, so slashes in `DisplayPath` values (e.g.
    /// `corpus/subdir/file.cha`) are safe.
    pub async fn get_file_result(
        &self,
        url: &str,
        job_id: &JobId,
        filename: &DisplayPath,
    ) -> Result<FileResult, CliError> {
        let resp = self
            .request_with_retry(
                reqwest::Method::GET,
                &format!("{url}/jobs/{job_id}/results/{filename}"),
                None::<&()>,
            )
            .await?;
        let result: FileResult = resp.json().await?;
        Ok(result)
    }

    /// `GET /jobs/{id}/results` — fetch all results for a job.
    pub async fn get_all_results(
        &self,
        url: &str,
        job_id: &JobId,
    ) -> Result<JobResultResponse, CliError> {
        let resp = self
            .request_with_retry(
                reqwest::Method::GET,
                &format!("{url}/jobs/{job_id}/results"),
                None::<&()>,
            )
            .await?;
        let results: JobResultResponse = resp.json().await?;
        Ok(results)
    }

    /// `GET /jobs` — list all jobs.
    pub async fn list_jobs(&self, url: &str) -> Result<Vec<JobListItem>, CliError> {
        let resp = self
            .http
            .get(format!("{url}/jobs"))
            .timeout(Duration::from_secs(10))
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let detail = read_http_error_detail(resp).await;
            return Err(CliError::ServerHttp { status, detail });
        }
        let jobs: Vec<JobListItem> = resp.json().await?;
        Ok(jobs)
    }

    /// `DELETE /jobs/{id}` — cancel a running job.
    pub async fn cancel_job(&self, url: &str, job_id: &JobId) -> Result<(), CliError> {
        let resp = self
            .http
            .delete(format!("{url}/jobs/{job_id}"))
            .timeout(Duration::from_secs(10))
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let detail = read_http_error_detail(resp).await;
            return Err(CliError::ServerHttp { status, detail });
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Retry logic
    // -----------------------------------------------------------------------

    /// HTTP request with exponential-backoff retry for transient errors.
    ///
    /// Retries on `ConnectionError`/`Timeout`, NOT on 4xx/5xx HTTP errors.
    async fn request_with_retry<B: serde::Serialize>(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<&B>,
    ) -> Result<reqwest::Response, CliError> {
        let mut delay = RETRY_BACKOFF;
        let mut last_err: Option<reqwest::Error> = None;

        for attempt in 0..RETRY_ATTEMPTS {
            let mut builder = self.http.request(method.clone(), url);
            if let Some(b) = body {
                builder = builder.json(b);
            }
            builder = builder.timeout(Duration::from_secs(30));

            match builder.send().await {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        let status = resp.status().as_u16();
                        let detail = read_http_error_detail(resp).await;
                        return Err(CliError::ServerHttp { status, detail });
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    let is_transient = e.is_connect() || e.is_timeout();
                    if is_transient && attempt < RETRY_ATTEMPTS - 1 {
                        debug!(
                            attempt = attempt + 1,
                            max = RETRY_ATTEMPTS,
                            url,
                            error = %e,
                            "Retrying transient error"
                        );
                        let jitter = 0.5 + rand::random::<f64>() * 0.5;
                        tokio::time::sleep(Duration::from_secs_f64(delay * jitter)).await;
                        delay *= 2.0;
                        last_err = Some(e);
                        continue;
                    }
                    return Err(e.into());
                }
            }
        }

        // SAFETY: the retry loop always sets last_err before `continue`
        Err(last_err
            .expect("retry loop exhausted without setting last_err")
            .into())
    }
}

impl Default for BatchalignClient {
    fn default() -> Self {
        Self::new()
    }
}

async fn read_http_error_detail(resp: reqwest::Response) -> String {
    let status = resp.status();
    let status_text = status.canonical_reason().unwrap_or("").to_string();
    match resp.text().await {
        Ok(body) => summarize_http_error_body(&status_text, &body),
        Err(error) if status_text.is_empty() => format!("failed to read error body: {error}"),
        Err(error) => format!("{status_text} (failed to read error body: {error})"),
    }
}

fn summarize_http_error_body(status_text: &str, body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return status_text.to_string();
    }

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed)
        && let Some(detail) = value.get("detail").and_then(|d| d.as_str())
        && !detail.trim().is_empty()
    {
        return detail.to_string();
    }

    trimmed.to_string()
}

/// Extract a short hostname label from a server URL (e.g. "myhost").
pub fn server_label(url: &str) -> String {
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    let host = without_scheme
        .split(':')
        .next()
        .unwrap_or(without_scheme)
        .split('/')
        .next()
        .unwrap_or(without_scheme);
    host.split('.').next().unwrap_or(host).to_string()
}

/// Parse comma-separated server URLs, strip whitespace and trailing slashes.
pub fn parse_servers(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_servers_basic() {
        let urls = parse_servers("http://a:8000, http://b:8000/ ,");
        assert_eq!(urls, vec!["http://a:8000", "http://b:8000"]);
    }

    #[test]
    fn server_label_extracts_hostname() {
        assert_eq!(server_label("http://net.local:8000"), "net");
        assert_eq!(server_label("https://192.168.1.1:8000/path"), "192");
        assert_eq!(server_label("http://myhost:9000"), "myhost");
    }

    #[test]
    fn job_status_is_terminal() {
        use batchalign_app::api::JobStatus;
        assert!(JobStatus::Completed.is_terminal());
        assert!(JobStatus::Failed.is_terminal());
        assert!(JobStatus::Cancelled.is_terminal());
        assert!(JobStatus::Interrupted.is_terminal());
        assert!(!JobStatus::Running.is_terminal());
        assert!(!JobStatus::Queued.is_terminal());
    }

    #[test]
    fn parse_servers_empty() {
        let urls = parse_servers("");
        assert!(urls.is_empty());
    }

    #[test]
    fn parse_servers_single() {
        let urls = parse_servers("http://a:8000");
        assert_eq!(urls, vec!["http://a:8000"]);
    }

    #[test]
    fn parse_servers_trailing_slashes() {
        let urls = parse_servers("http://a:8000///");
        assert_eq!(urls, vec!["http://a:8000"]);
    }

    #[test]
    fn summarize_http_error_body_prefers_json_detail() {
        let detail = summarize_http_error_body("Bad Request", r#"{"detail":"bad config"}"#);
        assert_eq!(detail, "bad config");
    }

    #[test]
    fn summarize_http_error_body_falls_back_to_raw_body() {
        let detail = summarize_http_error_body("Bad Request", "plain server failure");
        assert_eq!(detail, "plain server failure");
    }

    #[test]
    fn summarize_http_error_body_falls_back_to_status_text_when_empty() {
        let detail = summarize_http_error_body("Bad Request", "   ");
        assert_eq!(detail, "Bad Request");
    }

    #[test]
    fn server_label_no_scheme() {
        assert_eq!(server_label("myhost:9000"), "myhost");
        assert_eq!(server_label("bare-hostname"), "bare-hostname");
    }
}
