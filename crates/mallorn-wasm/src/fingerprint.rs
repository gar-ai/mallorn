//! Model fingerprinting for WASM

use wasm_bindgen::prelude::*;
use sha2::{Sha256, Digest};

/// Fingerprint result
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct Fingerprint {
    file_size: u64,
    header_hash: String,
    tail_hash: String,
    short_id: String,
}

#[wasm_bindgen]
impl Fingerprint {
    /// Get file size
    #[wasm_bindgen(getter)]
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Get header hash (hex)
    #[wasm_bindgen(getter)]
    pub fn header_hash(&self) -> String {
        self.header_hash.clone()
    }

    /// Get tail hash (hex)
    #[wasm_bindgen(getter)]
    pub fn tail_hash(&self) -> String {
        self.tail_hash.clone()
    }

    /// Get short identifier (first 16 chars of combined hash)
    #[wasm_bindgen(getter)]
    pub fn short_id(&self) -> String {
        self.short_id.clone()
    }
}

/// Generate a fingerprint from model data
///
/// Samples the first 64KB and last 4KB for quick version detection.
#[wasm_bindgen]
pub fn fingerprint(data: &[u8]) -> Fingerprint {
    const HEADER_SIZE: usize = 64 * 1024;
    const TAIL_SIZE: usize = 4 * 1024;

    let file_size = data.len() as u64;

    // Hash header
    let header_end = HEADER_SIZE.min(data.len());
    let mut header_hasher = Sha256::new();
    header_hasher.update(&data[..header_end]);
    let header_hash = hex::encode(&header_hasher.finalize()[..16]);

    // Hash tail
    let tail_start = if data.len() > TAIL_SIZE {
        data.len() - TAIL_SIZE
    } else {
        0
    };
    let mut tail_hasher = Sha256::new();
    tail_hasher.update(&data[tail_start..]);
    let tail_hash = hex::encode(&tail_hasher.finalize()[..16]);

    // Combine for short ID
    let short_id = format!("{}", &header_hash[..8]);

    Fingerprint {
        file_size,
        header_hash,
        tail_hash,
        short_id,
    }
}

/// Compare two fingerprints
#[wasm_bindgen]
pub fn fingerprints_match(a: &Fingerprint, b: &Fingerprint) -> bool {
    a.short_id == b.short_id && a.file_size == b.file_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_fingerprint() {
        let data = vec![0u8; 1024];
        let fp = fingerprint(&data);
        assert_eq!(fp.file_size, 1024);
        assert!(!fp.short_id.is_empty());
    }

    #[wasm_bindgen_test]
    fn test_fingerprint_match() {
        let data = vec![42u8; 2048];
        let fp1 = fingerprint(&data);
        let fp2 = fingerprint(&data);
        assert!(fingerprints_match(&fp1, &fp2));
    }
}
