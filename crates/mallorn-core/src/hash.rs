//! Hashing and verification utilities

use sha2::{Digest, Sha256};

/// Compute SHA256 hash of data
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Verify data matches expected SHA256 hash
pub fn verify_hash(data: &[u8], expected: &[u8; 32]) -> bool {
    sha256(data) == *expected
}

/// Format hash as hex string
pub fn hash_to_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Parse hex string to hash
pub fn hex_to_hash(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }

    let mut hash = [0u8; 32];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let s = std::str::from_utf8(chunk).ok()?;
        hash[i] = u8::from_str_radix(s, 16).ok()?;
    }
    Some(hash)
}

/// Compute CRC32 checksum (for patch integrity)
pub fn crc32(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

/// Verify CRC32 checksum
pub fn verify_crc32(data: &[u8], expected: u32) -> bool {
    crc32(data) == expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_deterministic() {
        let data = b"test data for hashing";
        let hash1 = sha256(data);
        let hash2 = sha256(data);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_sha256_different_input() {
        let hash1 = sha256(b"hello");
        let hash2 = sha256(b"world");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_verify_hash() {
        let data = b"test data";
        let hash = sha256(data);
        assert!(verify_hash(data, &hash));
        assert!(!verify_hash(b"other data", &hash));
    }

    #[test]
    fn test_hash_hex_roundtrip() {
        let data = b"test";
        let hash = sha256(data);
        let hex = hash_to_hex(&hash);
        let parsed = hex_to_hash(&hex).unwrap();
        assert_eq!(hash, parsed);
    }

    #[test]
    fn test_crc32() {
        let data = b"test data";
        let checksum = crc32(data);
        assert!(verify_crc32(data, checksum));
        assert!(!verify_crc32(b"other", checksum));
    }
}
