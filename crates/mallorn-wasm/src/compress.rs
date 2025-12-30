//! Compression utilities for WASM
//!
//! Uses pure-Rust implementations for WASM compatibility:
//! - `ruzstd` for zstd decompression (decode only)
//! - `lz4_flex` for lz4 compression/decompression

use crate::WasmError;
use wasm_bindgen::prelude::*;

/// Compress data using zstd
///
/// Note: In WASM, zstd compression is not available. Use `compress_lz4` instead.
/// This function is provided for API compatibility but will return an error.
#[wasm_bindgen]
pub fn compress_zstd(_data: &[u8], _level: i32) -> Result<Vec<u8>, WasmError> {
    Err(WasmError::new(
        "Zstd compression is not available in WASM. Use compress_lz4 instead.".to_string(),
    ))
}

/// Decompress zstd data using pure-Rust ruzstd decoder
#[wasm_bindgen]
pub fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>, WasmError> {
    use std::io::Read;

    let mut decoder = ruzstd::StreamingDecoder::new(data)
        .map_err(|e| WasmError::new(format!("Failed to create zstd decoder: {}", e)))?;

    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| WasmError::new(format!("Zstd decompression failed: {}", e)))?;

    Ok(decompressed)
}

/// Compress data using lz4
#[wasm_bindgen]
pub fn compress_lz4(data: &[u8]) -> Vec<u8> {
    lz4_flex::compress_prepend_size(data)
}

/// Decompress lz4 data
#[wasm_bindgen]
pub fn decompress_lz4(data: &[u8]) -> Result<Vec<u8>, WasmError> {
    lz4_flex::decompress_size_prepended(data)
        .map_err(|e| WasmError::new(format!("LZ4 decompression failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_zstd_compress_not_available() {
        // Zstd compression is not available in WASM
        let data = b"Hello, World!";
        let result = compress_zstd(data, 3);
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_zstd_decompress() {
        // Pre-compressed data using native zstd (level 3)
        // Original: b"Hello, World!"
        let compressed: &[u8] = &[
            0x28, 0xb5, 0x2f, 0xfd, 0x04, 0x00, 0x61, 0x00, 0x00, 0x48, 0x65, 0x6c, 0x6c, 0x6f,
            0x2c, 0x20, 0x57, 0x6f, 0x72, 0x6c, 0x64, 0x21, 0x5b, 0xc2, 0x58, 0xca,
        ];
        let decompressed = decompress_zstd(compressed).unwrap();
        assert_eq!(&decompressed, b"Hello, World!");
    }

    #[wasm_bindgen_test]
    fn test_lz4_roundtrip() {
        let data = b"Hello, World! This is a test of compression.";
        let compressed = compress_lz4(data);
        let decompressed = decompress_lz4(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }
}
