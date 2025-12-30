//! HTTP network utilities for patch downloads
//!
//! Provides HTTP range request support for efficient patch downloads,
//! resume capability, and header-only fetching.

use crate::error::PatchError;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Error types for network operations
#[derive(Debug, thiserror::Error)]
pub enum NetworkError {
    #[error("HTTP error: {status} - {message}")]
    HttpError { status: u16, message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Server does not support range requests")]
    RangeNotSupported,

    #[error("Download incomplete: expected {expected} bytes, got {got}")]
    IncompleteDownload { expected: u64, got: u64 },

    #[error("Hash mismatch: expected {expected}, got {got}")]
    HashMismatch { expected: String, got: String },

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Request failed: {0}")]
    RequestFailed(String),

    #[error("Parse error: {0}")]
    ParseError(String),
}

impl From<NetworkError> for PatchError {
    fn from(e: NetworkError) -> Self {
        PatchError::Io(std::io::Error::other(e.to_string()))
    }
}

/// Information about a remote patch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchInfo {
    /// URL to download the patch
    pub url: String,
    /// Source model hash (hex)
    pub source_hash: String,
    /// Target model hash (hex)
    pub target_hash: String,
    /// Patch size in bytes
    pub size: u64,
    /// Model format (tflite, gguf, etc.)
    pub format: String,
    /// Optional patch version/description
    pub description: Option<String>,
}

/// Information about a patch chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainInfo {
    /// Chain ID
    pub chain_id: String,
    /// URL to download the chain
    pub url: String,
    /// Base version hash
    pub base_hash: String,
    /// Head version hash
    pub head_hash: String,
    /// Number of patches in chain
    pub num_patches: usize,
    /// Total size of all patches
    pub total_size: u64,
}

/// Manifest of available patches for a model family
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchManifest {
    /// Model family name
    pub model_family: String,
    /// Available individual patches
    pub patches: Vec<PatchInfo>,
    /// Available patch chains
    pub chains: Vec<ChainInfo>,
    /// Manifest version
    pub version: u32,
    /// Last updated timestamp
    pub updated_at: u64,
}

impl PatchManifest {
    /// Create a new empty manifest
    pub fn new(model_family: &str) -> Self {
        Self {
            model_family: model_family.to_string(),
            patches: Vec::new(),
            chains: Vec::new(),
            version: 1,
            updated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Find patches that can update from a given source hash
    pub fn find_patches_from(&self, source_hash: &str) -> Vec<&PatchInfo> {
        self.patches
            .iter()
            .filter(|p| p.source_hash == source_hash)
            .collect()
    }

    /// Find a specific patch by source and target
    pub fn find_patch(&self, source_hash: &str, target_hash: &str) -> Option<&PatchInfo> {
        self.patches
            .iter()
            .find(|p| p.source_hash == source_hash && p.target_hash == target_hash)
    }

    /// Find chains that contain a specific version
    pub fn find_chains_containing(&self, hash: &str) -> Vec<&ChainInfo> {
        self.chains
            .iter()
            .filter(|c| c.base_hash == hash || c.head_hash == hash)
            .collect()
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Header information from a partial fetch
#[derive(Debug, Clone)]
pub struct PatchHeader {
    /// Total file size
    pub content_length: u64,
    /// Whether range requests are supported
    pub accepts_ranges: bool,
    /// Content type
    pub content_type: Option<String>,
    /// ETag for caching
    pub etag: Option<String>,
    /// Last modified time
    pub last_modified: Option<String>,
}

/// Progress callback for downloads
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send>;

/// Configuration for downloads
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Directory for temporary/partial downloads
    pub cache_dir: PathBuf,
    /// Whether to verify hash after download
    pub verify_hash: bool,
    /// Timeout in seconds for initial connection
    pub connect_timeout_secs: u64,
    /// Timeout in seconds for reading data
    pub read_timeout_secs: u64,
    /// Number of retry attempts
    pub retries: u32,
    /// Chunk size for range requests (bytes)
    pub chunk_size: usize,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            cache_dir: std::env::temp_dir().join("mallorn_cache"),
            verify_hash: true,
            connect_timeout_secs: 30,
            read_timeout_secs: 300,
            retries: 3,
            chunk_size: 1024 * 1024, // 1MB chunks
        }
    }
}

/// Download state for resumable downloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadState {
    /// Original URL
    pub url: String,
    /// Expected total size
    pub total_size: u64,
    /// Bytes downloaded so far
    pub downloaded: u64,
    /// ETag for validation
    pub etag: Option<String>,
    /// Path to partial file
    pub partial_path: String,
    /// Target file path
    pub target_path: String,
}

impl DownloadState {
    /// Create new download state
    pub fn new(url: &str, total_size: u64, partial_path: &Path, target_path: &Path) -> Self {
        Self {
            url: url.to_string(),
            total_size,
            downloaded: 0,
            etag: None,
            partial_path: partial_path.to_string_lossy().to_string(),
            target_path: target_path.to_string_lossy().to_string(),
        }
    }

    /// Calculate download progress (0.0 - 1.0)
    pub fn progress(&self) -> f64 {
        if self.total_size == 0 {
            0.0
        } else {
            self.downloaded as f64 / self.total_size as f64
        }
    }

    /// Check if download is complete
    pub fn is_complete(&self) -> bool {
        self.downloaded >= self.total_size
    }

    /// Save state to file
    pub fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Load state from file
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let json = fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Simulated download for local file URLs (file://)
///
/// This is useful for testing and local development.
pub fn download_local_file(url: &str, target: &Path) -> Result<u64, NetworkError> {
    let path = url
        .strip_prefix("file://")
        .ok_or_else(|| NetworkError::InvalidUrl(url.to_string()))?;

    let source = Path::new(path);
    if !source.exists() {
        return Err(NetworkError::RequestFailed(format!(
            "File not found: {}",
            path
        )));
    }

    fs::copy(source, target)?;
    let size = fs::metadata(target)?.len();
    Ok(size)
}

/// Check if a URL supports range requests by doing a HEAD request
///
/// Returns header information including whether ranges are supported.
pub fn check_range_support(url: &str) -> Result<PatchHeader, NetworkError> {
    // For now, return a placeholder that indicates no range support
    // Real implementation would use reqwest
    if url.starts_with("file://") {
        let path = url
            .strip_prefix("file://")
            .ok_or_else(|| NetworkError::InvalidUrl(url.to_string()))?;
        let metadata = fs::metadata(path)?;
        return Ok(PatchHeader {
            content_length: metadata.len(),
            accepts_ranges: false, // Local files don't need range requests
            content_type: Some("application/octet-stream".to_string()),
            etag: None,
            last_modified: None,
        });
    }

    // Placeholder for HTTP URLs - real implementation would use reqwest
    Err(NetworkError::RequestFailed(
        "HTTP support requires 'network' feature".to_string(),
    ))
}

/// Calculate SHA256 hash of a file
pub fn hash_file(path: &Path) -> Result<[u8; 32], std::io::Error> {
    use sha2::{Digest, Sha256};

    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hasher.finalize().into())
}

/// Verify a downloaded file matches expected hash
pub fn verify_download(path: &Path, expected_hash: &[u8; 32]) -> Result<bool, std::io::Error> {
    let actual_hash = hash_file(path)?;
    Ok(&actual_hash == expected_hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_patch_manifest_creation() {
        let manifest = PatchManifest::new("llama-7b");
        assert_eq!(manifest.model_family, "llama-7b");
        assert!(manifest.patches.is_empty());
        assert!(manifest.chains.is_empty());
    }

    #[test]
    fn test_patch_manifest_json_roundtrip() {
        let mut manifest = PatchManifest::new("test-model");
        manifest.patches.push(PatchInfo {
            url: "https://example.com/patch1.tflp".to_string(),
            source_hash: "aabbccdd".to_string(),
            target_hash: "11223344".to_string(),
            size: 1024,
            format: "tflite".to_string(),
            description: Some("Test patch".to_string()),
        });

        let json = manifest.to_json().unwrap();
        let restored = PatchManifest::from_json(&json).unwrap();

        assert_eq!(restored.model_family, manifest.model_family);
        assert_eq!(restored.patches.len(), 1);
        assert_eq!(restored.patches[0].url, "https://example.com/patch1.tflp");
    }

    #[test]
    fn test_find_patches() {
        let mut manifest = PatchManifest::new("test");
        manifest.patches.push(PatchInfo {
            url: "patch1".to_string(),
            source_hash: "aabb".to_string(),
            target_hash: "ccdd".to_string(),
            size: 100,
            format: "tflite".to_string(),
            description: None,
        });
        manifest.patches.push(PatchInfo {
            url: "patch2".to_string(),
            source_hash: "aabb".to_string(),
            target_hash: "eeff".to_string(),
            size: 200,
            format: "tflite".to_string(),
            description: None,
        });

        let patches = manifest.find_patches_from("aabb");
        assert_eq!(patches.len(), 2);

        let specific = manifest.find_patch("aabb", "ccdd");
        assert!(specific.is_some());
        assert_eq!(specific.unwrap().url, "patch1");
    }

    #[test]
    fn test_download_state() {
        let state = DownloadState::new(
            "https://example.com/patch.bin",
            1000,
            Path::new("/tmp/partial"),
            Path::new("/tmp/complete"),
        );

        assert_eq!(state.progress(), 0.0);
        assert!(!state.is_complete());

        let mut state = state;
        state.downloaded = 500;
        assert!((state.progress() - 0.5).abs() < 0.01);

        state.downloaded = 1000;
        assert!(state.is_complete());
    }

    #[test]
    fn test_download_state_save_load() {
        let dir = tempdir().unwrap();
        let state_path = dir.path().join("download.state");

        let state = DownloadState {
            url: "https://example.com/test".to_string(),
            total_size: 5000,
            downloaded: 2500,
            etag: Some("abc123".to_string()),
            partial_path: "/tmp/partial.bin".to_string(),
            target_path: "/tmp/final.bin".to_string(),
        };

        state.save(&state_path).unwrap();
        let loaded = DownloadState::load(&state_path).unwrap();

        assert_eq!(loaded.url, state.url);
        assert_eq!(loaded.downloaded, state.downloaded);
        assert_eq!(loaded.etag, state.etag);
    }

    #[test]
    fn test_download_config_defaults() {
        let config = DownloadConfig::default();
        assert!(config.verify_hash);
        assert_eq!(config.retries, 3);
        assert_eq!(config.chunk_size, 1024 * 1024);
    }

    #[test]
    fn test_download_local_file() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.bin");
        let target = dir.path().join("target.bin");

        // Create source file
        {
            let mut f = File::create(&source).unwrap();
            f.write_all(b"test data content").unwrap();
        }

        let url = format!("file://{}", source.display());
        let size = download_local_file(&url, &target).unwrap();

        assert!(target.exists());
        assert_eq!(size, 17);
    }

    #[test]
    fn test_hash_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.bin");

        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"hello world").unwrap();
        }

        let hash = hash_file(&path).unwrap();
        // Known SHA256 of "hello world"
        let expected = hex::decode("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9")
            .unwrap();
        assert_eq!(hash.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_verify_download() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.bin");

        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"hello world").unwrap();
        }

        // Correct hash
        let correct_hash: [u8; 32] = hex::decode(
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        )
        .unwrap()
        .try_into()
        .unwrap();
        assert!(verify_download(&path, &correct_hash).unwrap());

        // Wrong hash
        let wrong_hash = [0u8; 32];
        assert!(!verify_download(&path, &wrong_hash).unwrap());
    }

    #[test]
    fn test_check_range_support_local() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.bin");

        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&[0u8; 1000]).unwrap();
        }

        let url = format!("file://{}", path.display());
        let header = check_range_support(&url).unwrap();

        assert_eq!(header.content_length, 1000);
        assert!(!header.accepts_ranges); // Local files don't use ranges
    }
}
