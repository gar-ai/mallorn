//! Async HTTP network utilities for patch downloads
//!
//! Provides async HTTP range request support for non-blocking patch downloads.
//!
//! # Example
//!
//! ```ignore
//! use mallorn_core::network_async::download_async;
//!
//! #[tokio::main]
//! async fn main() {
//!     let bytes = download_async("https://example.com/patch.tflp", None).await?;
//!     println!("Downloaded {} bytes", bytes.len());
//! }
//! ```

use crate::network::{DownloadState, NetworkError};
use std::path::Path;

/// Async download configuration
#[derive(Debug, Clone)]
pub struct AsyncDownloadConfig {
    /// Timeout for connection in milliseconds
    pub connect_timeout_ms: u64,
    /// Timeout for read operations in milliseconds
    pub read_timeout_ms: u64,
    /// Buffer size for streaming downloads
    pub buffer_size: usize,
    /// Whether to verify hash after download
    pub verify_hash: bool,
}

impl Default for AsyncDownloadConfig {
    fn default() -> Self {
        Self {
            connect_timeout_ms: 30_000,
            read_timeout_ms: 60_000,
            buffer_size: 64 * 1024,
            verify_hash: true,
        }
    }
}

/// Progress callback for async downloads
pub trait AsyncProgress: Send + Sync {
    /// Called periodically with download progress
    fn on_progress(&self, downloaded: u64, total: Option<u64>);

    /// Called when download is complete
    fn on_complete(&self, total_bytes: u64);

    /// Called on error
    fn on_error(&self, error: &str);
}

/// Simple progress reporter that does nothing
pub struct NoopProgress;

impl AsyncProgress for NoopProgress {
    fn on_progress(&self, _downloaded: u64, _total: Option<u64>) {}
    fn on_complete(&self, _total_bytes: u64) {}
    fn on_error(&self, _error: &str) {}
}

/// Download a file asynchronously with resume support
///
/// This is a placeholder implementation. The actual async HTTP client
/// would use reqwest with the `stream` feature enabled.
#[cfg(feature = "async")]
pub async fn download_async(
    url: &str,
    output: &Path,
    state: Option<&DownloadState>,
    config: &AsyncDownloadConfig,
    progress: &dyn AsyncProgress,
) -> Result<u64, NetworkError> {
    use tokio::io::AsyncWriteExt;

    // Build async HTTP client
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_millis(config.connect_timeout_ms))
        .timeout(std::time::Duration::from_millis(config.read_timeout_ms))
        .build()
        .map_err(|e| NetworkError::ConnectionFailed(e.to_string()))?;

    // Check for resume
    let start_byte = state.map(|s| s.downloaded).unwrap_or(0);

    // Build request with range header if resuming
    let mut request = client.get(url);
    if start_byte > 0 {
        request = request.header("Range", format!("bytes={}-", start_byte));
    }

    // Send request
    let response = request
        .send()
        .await
        .map_err(|e| NetworkError::ConnectionFailed(e.to_string()))?;

    if !response.status().is_success() && response.status().as_u16() != 206 {
        return Err(NetworkError::HttpError {
            status: response.status().as_u16(),
            message: response.status().to_string(),
        });
    }

    // Get content length
    let content_length = response.content_length();

    // Open file for writing
    let mut file = if start_byte > 0 {
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(output)
            .await
            .map_err(|e| NetworkError::IoError(e.to_string()))?
    } else {
        tokio::fs::File::create(output)
            .await
            .map_err(|e| NetworkError::IoError(e.to_string()))?
    };

    // Stream response body
    let mut bytes_written = start_byte;
    let mut stream = response.bytes_stream();

    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| NetworkError::IoError(e.to_string()))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| NetworkError::IoError(e.to_string()))?;
        bytes_written += chunk.len() as u64;
        progress.on_progress(bytes_written, content_length);
    }

    progress.on_complete(bytes_written);

    Ok(bytes_written)
}

/// Download multiple files concurrently
#[cfg(feature = "async")]
pub async fn download_many_async(
    downloads: Vec<(&str, &Path)>,
    config: &AsyncDownloadConfig,
) -> Vec<Result<u64, NetworkError>> {
    use futures_util::future::join_all;

    let progress = NoopProgress;
    let futures: Vec<_> = downloads
        .into_iter()
        .map(|(url, path)| download_async(url, path, None, config, &progress))
        .collect();

    join_all(futures).await
}

/// Check if server supports range requests asynchronously
#[cfg(feature = "async")]
pub async fn check_range_support_async(url: &str) -> Result<bool, NetworkError> {
    let client = reqwest::Client::new();

    let response = client
        .head(url)
        .send()
        .await
        .map_err(|e| NetworkError::ConnectionFailed(e.to_string()))?;

    let accepts_ranges = response
        .headers()
        .get("accept-ranges")
        .map(|v: &reqwest::header::HeaderValue| v.to_str().unwrap_or("") == "bytes")
        .unwrap_or(false);

    Ok(accepts_ranges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = AsyncDownloadConfig::default();
        assert_eq!(config.connect_timeout_ms, 30_000);
        assert_eq!(config.read_timeout_ms, 60_000);
        assert_eq!(config.buffer_size, 64 * 1024);
        assert!(config.verify_hash);
    }
}
