//! Delta computation for WASM

use wasm_bindgen::prelude::*;
use crate::WasmError;

/// Compute XOR delta between two byte arrays
#[wasm_bindgen]
pub fn compute_delta(old: &[u8], new: &[u8]) -> Result<Vec<u8>, WasmError> {
    if old.len() != new.len() {
        return Err(WasmError::new(format!(
            "Size mismatch: old={}, new={}",
            old.len(),
            new.len()
        )));
    }

    Ok(old.iter().zip(new.iter()).map(|(a, b)| a ^ b).collect())
}

/// Apply XOR delta to reconstruct new data
#[wasm_bindgen]
pub fn apply_delta(old: &[u8], delta: &[u8]) -> Result<Vec<u8>, WasmError> {
    if old.len() != delta.len() {
        return Err(WasmError::new(format!(
            "Size mismatch: old={}, delta={}",
            old.len(),
            delta.len()
        )));
    }

    // XOR is its own inverse
    Ok(old.iter().zip(delta.iter()).map(|(a, b)| a ^ b).collect())
}

/// Statistics about a delta
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DeltaStats {
    total_bytes: usize,
    changed_bytes: usize,
    zero_bytes: usize,
}

#[wasm_bindgen]
impl DeltaStats {
    /// Total bytes in delta
    #[wasm_bindgen(getter)]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Number of non-zero bytes (actual changes)
    #[wasm_bindgen(getter)]
    pub fn changed_bytes(&self) -> usize {
        self.changed_bytes
    }

    /// Number of zero bytes (unchanged)
    #[wasm_bindgen(getter)]
    pub fn zero_bytes(&self) -> usize {
        self.zero_bytes
    }

    /// Percentage of bytes that changed
    #[wasm_bindgen(getter)]
    pub fn change_rate(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.changed_bytes as f64 / self.total_bytes as f64
        }
    }
}

/// Analyze a delta to get statistics
#[wasm_bindgen]
pub fn analyze_delta(delta: &[u8]) -> DeltaStats {
    let zero_bytes = delta.iter().filter(|&&b| b == 0).count();
    let changed_bytes = delta.len() - zero_bytes;

    DeltaStats {
        total_bytes: delta.len(),
        changed_bytes,
        zero_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_delta_roundtrip() {
        let old = vec![1, 2, 3, 4, 5];
        let new = vec![1, 3, 3, 5, 5];

        let delta = compute_delta(&old, &new).unwrap();
        let restored = apply_delta(&old, &delta).unwrap();

        assert_eq!(restored, new);
    }

    #[wasm_bindgen_test]
    fn test_delta_stats() {
        let delta = vec![0, 1, 0, 2, 0, 0, 3, 0];
        let stats = analyze_delta(&delta);

        assert_eq!(stats.total_bytes, 8);
        assert_eq!(stats.changed_bytes, 3);
        assert_eq!(stats.zero_bytes, 5);
    }
}
