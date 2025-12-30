//! Patch application for WASM

use crate::compress::{compress_lz4, decompress_lz4, decompress_zstd};
use crate::WasmError;
use sha2::{Digest, Sha256};
use wasm_bindgen::prelude::*;

/// Patch application result
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct PatchResult {
    data: Vec<u8>,
    source_hash: String,
    target_hash: String,
    success: bool,
}

#[wasm_bindgen]
impl PatchResult {
    /// Get the resulting model data
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Get source model hash
    #[wasm_bindgen(getter)]
    pub fn source_hash(&self) -> String {
        self.source_hash.clone()
    }

    /// Get target model hash
    #[wasm_bindgen(getter)]
    pub fn target_hash(&self) -> String {
        self.target_hash.clone()
    }

    /// Check if patch was successful
    #[wasm_bindgen(getter)]
    pub fn success(&self) -> bool {
        self.success
    }
}

/// Apply a simple XOR patch to a model
///
/// This is a simplified patch format for WASM:
/// - First 32 bytes: expected source hash
/// - Next 32 bytes: expected target hash
/// - Remaining: compressed XOR delta
#[wasm_bindgen]
pub fn apply_patch(model: &[u8], patch: &[u8]) -> Result<PatchResult, WasmError> {
    if patch.len() < 64 {
        return Err(WasmError::new("Patch too small"));
    }

    // Extract hashes
    let expected_source_hash = &patch[0..32];
    let expected_target_hash = &patch[32..64];
    let compressed_delta = &patch[64..];

    // Verify source hash
    let mut hasher = Sha256::new();
    hasher.update(model);
    let actual_source_hash = hasher.finalize();

    if actual_source_hash.as_slice() != expected_source_hash {
        return Err(WasmError::new("Source hash mismatch - wrong model version"));
    }

    // Decompress delta (supports both zstd and lz4)
    // Try zstd first (ruzstd pure-Rust decoder), then lz4
    let delta = decompress_zstd(compressed_delta)
        .or_else(|_| decompress_lz4(compressed_delta))
        .map_err(|e| WasmError::new(format!("Failed to decompress patch: {}", e)))?;

    if delta.len() != model.len() {
        return Err(WasmError::new(format!(
            "Delta size mismatch: model={}, delta={}",
            model.len(),
            delta.len()
        )));
    }

    // Apply delta
    let new_model: Vec<u8> = model.iter().zip(delta.iter()).map(|(a, b)| a ^ b).collect();

    // Verify target hash
    let mut hasher = Sha256::new();
    hasher.update(&new_model);
    let actual_target_hash = hasher.finalize();

    let success = actual_target_hash.as_slice() == expected_target_hash;

    Ok(PatchResult {
        data: new_model,
        source_hash: hex::encode(expected_source_hash),
        target_hash: hex::encode(actual_target_hash),
        success,
    })
}

/// Create a simple patch (for testing/development)
///
/// Note: In WASM, patches are compressed with LZ4 (pure-Rust).
/// The compression_level parameter is ignored in WASM builds.
#[wasm_bindgen]
pub fn create_patch(
    old_model: &[u8],
    new_model: &[u8],
    _compression_level: i32,
) -> Result<Vec<u8>, WasmError> {
    if old_model.len() != new_model.len() {
        return Err(WasmError::new("Model sizes must match"));
    }

    // Compute hashes
    let mut hasher = Sha256::new();
    hasher.update(old_model);
    let source_hash = hasher.finalize();

    let mut hasher = Sha256::new();
    hasher.update(new_model);
    let target_hash = hasher.finalize();

    // Compute delta
    let delta: Vec<u8> = old_model
        .iter()
        .zip(new_model.iter())
        .map(|(a, b)| a ^ b)
        .collect();

    // Compress delta using LZ4 (pure-Rust, WASM-compatible)
    let compressed_delta = compress_lz4(&delta);

    // Build patch
    let mut patch = Vec::with_capacity(64 + compressed_delta.len());
    patch.extend_from_slice(&source_hash);
    patch.extend_from_slice(&target_hash);
    patch.extend_from_slice(&compressed_delta);

    Ok(patch)
}

/// Get patch information without applying
#[wasm_bindgen]
pub fn patch_info(patch: &[u8]) -> Result<String, WasmError> {
    if patch.len() < 64 {
        return Err(WasmError::new("Patch too small"));
    }

    let source_hash = hex::encode(&patch[0..32]);
    let target_hash = hex::encode(&patch[32..64]);
    let compressed_size = patch.len() - 64;

    Ok(serde_json::json!({
        "source_hash": source_hash,
        "target_hash": target_hash,
        "compressed_size": compressed_size
    })
    .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_patch_roundtrip() {
        let old = vec![0u8; 1024];
        let mut new = vec![0u8; 1024];
        new[100] = 42;
        new[500] = 123;

        let patch = create_patch(&old, &new, 3).unwrap();
        let result = apply_patch(&old, &patch).unwrap();

        assert!(result.success);
        assert_eq!(result.data, new);
    }
}
