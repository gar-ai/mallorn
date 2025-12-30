//! `mallorn download` command implementation

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use mallorn_core::network::{
    check_range_support, download_local_file, verify_download, DownloadState, PatchManifest,
};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Download a patch from URL with resume support
pub fn run(
    url: &str,
    output: &Path,
    resume: bool,
    verify_hash: Option<&str>,
    cache_dir: Option<&Path>,
) -> Result<()> {
    println!("Downloading patch...");
    println!("  URL: {}", url);
    println!("  Output: {}", output.display());
    if resume {
        println!("  Resume: enabled");
    }

    // Parse expected hash if provided
    let expected_hash = if let Some(hash_str) = verify_hash {
        let bytes = hex::decode(hash_str).context("Invalid hash format (expected hex)")?;
        if bytes.len() != 32 {
            anyhow::bail!("Hash must be 32 bytes (64 hex characters)");
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Some(arr)
    } else {
        None
    };

    // Handle file:// URLs (local files)
    if url.starts_with("file://") {
        let size = download_local_file(url, output)
            .map_err(|e| anyhow::anyhow!("Download failed: {}", e))?;

        // Verify if requested
        if let Some(expected) = expected_hash {
            if !verify_download(output, &expected).context("Hash verification failed")? {
                anyhow::bail!("Hash mismatch!");
            }
            println!("  Hash: verified");
        }

        println!();
        println!("Download complete!");
        println!("  Size: {} bytes", size);
        return Ok(());
    }

    // HTTP/HTTPS download with reqwest
    if url.starts_with("http://") || url.starts_with("https://") {
        return download_http(url, output, resume, expected_hash, cache_dir);
    }

    anyhow::bail!("Unsupported URL scheme. Use file://, http://, or https://");
}

/// Download via HTTP with resume support and progress bar
fn download_http(
    url: &str,
    output: &Path,
    resume: bool,
    expected_hash: Option<[u8; 32]>,
    cache_dir: Option<&Path>,
) -> Result<()> {
    // Set up cache directory for download state
    let cache = cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::temp_dir().join("mallorn-cache"));

    fs::create_dir_all(&cache).context("Failed to create cache directory")?;

    let state_file = cache.join(format!(
        "download_{}.state",
        hex::encode(&sha256_string(url)[..8])
    ));

    // Check for existing partial download
    let mut start_byte: u64 = 0;
    if resume && output.exists() {
        if let Ok(metadata) = fs::metadata(output) {
            start_byte = metadata.len();
            println!("  Resuming from {} bytes", start_byte);
        }
    }

    // Build HTTP client
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .context("Failed to create HTTP client")?;

    // First, get content length with HEAD request
    let head_response = client.head(url).send().context("Failed to fetch headers")?;

    if !head_response.status().is_success() {
        anyhow::bail!("HTTP error: {}", head_response.status());
    }

    let content_length = head_response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    let accepts_ranges = head_response
        .headers()
        .get(reqwest::header::ACCEPT_RANGES)
        .map(|v| v.to_str().unwrap_or("") == "bytes")
        .unwrap_or(false);

    // If we have a partial download but server doesn't support ranges, start fresh
    if start_byte > 0 && !accepts_ranges {
        println!("  Server doesn't support resume, starting fresh");
        start_byte = 0;
    }

    // If already complete, just verify
    if start_byte >= content_length && content_length > 0 {
        println!("  Download already complete");
        if let Some(expected) = expected_hash {
            if !verify_download(output, &expected).context("Hash verification failed")? {
                anyhow::bail!("Hash mismatch!");
            }
            println!("  Hash: verified");
        }
        return Ok(());
    }

    // Build download request with Range header if resuming
    let mut request = client.get(url);
    if start_byte > 0 && accepts_ranges {
        request = request.header(reqwest::header::RANGE, format!("bytes={}-", start_byte));
    }

    let response = request.send().context("Failed to start download")?;

    if !response.status().is_success() && response.status() != reqwest::StatusCode::PARTIAL_CONTENT
    {
        anyhow::bail!("HTTP error: {}", response.status());
    }

    // Set up progress bar
    let total_size = if start_byte > 0 {
        content_length - start_byte
    } else {
        content_length
    };

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    // Open output file (append if resuming)
    let mut file = if start_byte > 0 {
        fs::OpenOptions::new()
            .append(true)
            .open(output)
            .context("Failed to open output file for append")?
    } else {
        File::create(output).context("Failed to create output file")?
    };

    // Save download state
    let output_str = output.display().to_string();
    let state = DownloadState {
        url: url.to_string(),
        downloaded: start_byte,
        total_size: content_length,
        etag: head_response
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string()),
        partial_path: output_str.clone(),
        target_path: output_str.clone(),
    };
    state.save(&state_file).ok(); // Ignore save errors

    // Download with progress
    let mut downloaded: u64 = 0;
    let mut reader = response;
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = std::io::Read::read(&mut reader, &mut buffer)
            .context("Failed to read from response")?;
        if bytes_read == 0 {
            break;
        }

        file.write_all(&buffer[..bytes_read])
            .context("Failed to write to output file")?;
        downloaded += bytes_read as u64;
        pb.set_position(downloaded);

        // Update state periodically (every 1MB)
        #[allow(clippy::manual_is_multiple_of)]
        if downloaded % (1024 * 1024) == 0 {
            let updated_state = DownloadState {
                url: url.to_string(),
                downloaded: start_byte + downloaded,
                total_size: content_length,
                etag: state.etag.clone(),
                partial_path: state.partial_path.clone(),
                target_path: state.target_path.clone(),
            };
            updated_state.save(&state_file).ok();
        }
    }

    pb.finish_with_message("Download complete");

    // Remove state file on success
    fs::remove_file(&state_file).ok();

    // Verify hash if provided
    if let Some(expected) = expected_hash {
        print!("Verifying hash... ");
        if !verify_download(output, &expected).context("Hash verification failed")? {
            anyhow::bail!("Hash mismatch!");
        }
        println!("verified");
    }

    println!();
    println!("Download complete!");
    println!("  Size: {} bytes", start_byte + downloaded);

    Ok(())
}

/// Fetch header only (for inspection)
pub fn header(url: &str) -> Result<()> {
    println!("Fetching patch header...");
    println!("  URL: {}", url);

    // Use core's check_range_support for file:// URLs
    if url.starts_with("file://") {
        let header = check_range_support(url)
            .map_err(|e| anyhow::anyhow!("Failed to fetch header: {}", e))?;

        println!();
        println!("Patch Header");
        println!("============");
        println!("Content-Length: {} bytes", header.content_length);
        println!("Accepts-Ranges: {}", header.accepts_ranges);
        if let Some(ct) = header.content_type {
            println!("Content-Type:   {}", ct);
        }
        if let Some(etag) = header.etag {
            println!("ETag:           {}", etag);
        }
        if let Some(lm) = header.last_modified {
            println!("Last-Modified:  {}", lm);
        }
        return Ok(());
    }

    // HTTP header fetch with reqwest
    let client = reqwest::blocking::Client::new();
    let response = client.head(url).send().context("Failed to fetch headers")?;

    if !response.status().is_success() {
        anyhow::bail!("HTTP error: {}", response.status());
    }

    println!();
    println!("Patch Header");
    println!("============");

    if let Some(len) = response.headers().get(reqwest::header::CONTENT_LENGTH) {
        println!("Content-Length: {} bytes", len.to_str().unwrap_or("?"));
    }

    let accepts_ranges = response
        .headers()
        .get(reqwest::header::ACCEPT_RANGES)
        .map(|v| v.to_str().unwrap_or("") == "bytes")
        .unwrap_or(false);
    println!("Accepts-Ranges: {}", accepts_ranges);

    if let Some(ct) = response.headers().get(reqwest::header::CONTENT_TYPE) {
        println!("Content-Type:   {}", ct.to_str().unwrap_or("?"));
    }

    if let Some(etag) = response.headers().get(reqwest::header::ETAG) {
        println!("ETag:           {}", etag.to_str().unwrap_or("?"));
    }

    if let Some(lm) = response.headers().get(reqwest::header::LAST_MODIFIED) {
        println!("Last-Modified:  {}", lm.to_str().unwrap_or("?"));
    }

    Ok(())
}

/// Fetch and display a patch manifest
pub fn manifest(url: &str, json_output: bool) -> Result<()> {
    println!("Fetching patch manifest...");
    println!("  URL: {}", url);

    let data = if url.starts_with("file://") {
        // Local file
        let path = url
            .strip_prefix("file://")
            .ok_or_else(|| anyhow::anyhow!("Invalid file URL"))?;
        fs::read_to_string(path).context("Failed to read manifest file")?
    } else {
        // HTTP fetch
        let client = reqwest::blocking::Client::new();
        let response = client.get(url).send().context("Failed to fetch manifest")?;

        if !response.status().is_success() {
            anyhow::bail!("HTTP error: {}", response.status());
        }

        response.text().context("Failed to read response body")?
    };

    let manifest: PatchManifest =
        serde_json::from_str(&data).context("Failed to parse manifest JSON")?;

    if json_output {
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }

    println!();
    println!("Patch Manifest: {}", manifest.model_family);
    println!("================");
    println!("Version:    {}", manifest.version);
    println!("Patches:    {}", manifest.patches.len());
    println!("Chains:     {}", manifest.chains.len());

    if !manifest.patches.is_empty() {
        println!();
        println!("Available Patches:");
        for patch in &manifest.patches {
            println!(
                "  {} -> {} ({} bytes, {})",
                &patch.source_hash[..16.min(patch.source_hash.len())],
                &patch.target_hash[..16.min(patch.target_hash.len())],
                patch.size,
                patch.format
            );
        }
    }

    if !manifest.chains.is_empty() {
        println!();
        println!("Available Chains:");
        for chain in &manifest.chains {
            println!("  {} ({} patches)", chain.chain_id, chain.num_patches);
        }
    }

    Ok(())
}

fn sha256_string(s: &str) -> [u8; 32] {
    mallorn_core::sha256(s.as_bytes())
}
