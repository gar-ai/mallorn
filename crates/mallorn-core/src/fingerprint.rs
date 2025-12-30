//! Model fingerprinting for quick version detection
//!
//! Provides fast (~10ms) fingerprinting of model files without reading
//! the entire file. Uses header/tail hashing plus file size to create
//! a unique fingerprint that can identify model versions.

use crate::error::ParseError;
use md5::{Digest, Md5};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Size of header to hash (64KB)
const HEADER_SIZE: u64 = 64 * 1024;
/// Size of tail to hash (4KB)
const TAIL_SIZE: u64 = 4 * 1024;

/// Error types for fingerprinting operations
#[derive(Debug, thiserror::Error)]
pub enum FingerprintError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("File too small for fingerprinting")]
    FileTooSmall,

    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("Invalid fingerprint string")]
    InvalidFormat,
}

/// Quick fingerprint from model header and tail
///
/// Designed for fast (~10ms) model identification without reading
/// the entire file. Combines:
/// - MD5 hash of first 64KB (captures format header and early tensors)
/// - MD5 hash of last 4KB (captures final tensors and footer)
/// - File size (for disambiguation)
/// - Tensor count (if available from metadata)
/// - Format string
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelFingerprint {
    /// Model format (e.g., "tflite", "gguf", "onnx")
    pub format: String,
    /// MD5 hash of first 64KB
    pub header_hash: [u8; 16],
    /// MD5 hash of last 4KB
    pub tail_hash: [u8; 16],
    /// Total file size in bytes
    pub file_size: u64,
    /// Number of tensors (if known)
    pub tensor_count: Option<u32>,
    /// Model version from metadata (if available)
    pub metadata_version: Option<String>,
}

impl ModelFingerprint {
    /// Generate fingerprint from a file path
    ///
    /// This is very fast (~10ms) regardless of file size because
    /// it only reads the header and tail of the file.
    pub fn from_file(path: &Path) -> Result<Self, FingerprintError> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len();

        Self::from_reader(file, file_size)
    }

    /// Generate fingerprint from a reader with known size
    pub fn from_reader<R: Read + Seek>(
        mut reader: R,
        file_size: u64,
    ) -> Result<Self, FingerprintError> {
        if file_size < HEADER_SIZE + TAIL_SIZE {
            // For small files, hash the entire content
            return Self::from_small_reader(reader, file_size);
        }

        // Hash header (first 64KB)
        let mut header_buf = vec![0u8; HEADER_SIZE as usize];
        reader.seek(SeekFrom::Start(0))?;
        reader.read_exact(&mut header_buf)?;
        let header_hash = compute_md5(&header_buf);

        // Hash tail (last 4KB)
        let mut tail_buf = vec![0u8; TAIL_SIZE as usize];
        reader.seek(SeekFrom::End(-(TAIL_SIZE as i64)))?;
        reader.read_exact(&mut tail_buf)?;
        let tail_hash = compute_md5(&tail_buf);

        // Detect format from header magic bytes
        let format = detect_format(&header_buf);

        Ok(Self {
            format,
            header_hash,
            tail_hash,
            file_size,
            tensor_count: None,
            metadata_version: None,
        })
    }

    /// Generate fingerprint from a small file (read entire content)
    fn from_small_reader<R: Read + Seek>(
        mut reader: R,
        file_size: u64,
    ) -> Result<Self, FingerprintError> {
        if file_size < 4 {
            return Err(FingerprintError::FileTooSmall);
        }

        let mut buffer = vec![0u8; file_size as usize];
        reader.seek(SeekFrom::Start(0))?;
        reader.read_exact(&mut buffer)?;

        let hash = compute_md5(&buffer);
        let format = detect_format(&buffer);

        Ok(Self {
            format,
            header_hash: hash,
            tail_hash: hash, // Same for small files
            file_size,
            tensor_count: None,
            metadata_version: None,
        })
    }

    /// Generate fingerprint from in-memory data
    pub fn from_bytes(data: &[u8]) -> Result<Self, FingerprintError> {
        let file_size = data.len() as u64;

        if file_size < 4 {
            return Err(FingerprintError::FileTooSmall);
        }

        if file_size < HEADER_SIZE + TAIL_SIZE {
            let hash = compute_md5(data);
            let format = detect_format(data);

            return Ok(Self {
                format,
                header_hash: hash,
                tail_hash: hash,
                file_size,
                tensor_count: None,
                metadata_version: None,
            });
        }

        let header_hash = compute_md5(&data[..HEADER_SIZE as usize]);
        let tail_hash = compute_md5(&data[data.len() - TAIL_SIZE as usize..]);
        let format = detect_format(data);

        Ok(Self {
            format,
            header_hash,
            tail_hash,
            file_size,
            tensor_count: None,
            metadata_version: None,
        })
    }

    /// Set the tensor count (typically from parsing metadata)
    pub fn with_tensor_count(mut self, count: u32) -> Self {
        self.tensor_count = Some(count);
        self
    }

    /// Set the metadata version
    pub fn with_version(mut self, version: String) -> Self {
        self.metadata_version = Some(version);
        self
    }

    /// Check if this fingerprint matches another
    ///
    /// Matches if header, tail, and size all match.
    /// Format and metadata are not required to match.
    pub fn matches(&self, other: &Self) -> bool {
        self.header_hash == other.header_hash
            && self.tail_hash == other.tail_hash
            && self.file_size == other.file_size
    }

    /// Generate a compact string representation
    ///
    /// Format: `{format}:{header_hex}:{tail_hex}:{size}`
    pub fn to_compact_string(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.format,
            hex::encode(self.header_hash),
            hex::encode(self.tail_hash),
            self.file_size
        )
    }

    /// Parse from compact string representation
    pub fn from_compact_string(s: &str) -> Result<Self, FingerprintError> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 4 {
            return Err(FingerprintError::InvalidFormat);
        }

        let format = parts[0].to_string();
        let header_hash = hex::decode(parts[1])
            .map_err(|_| FingerprintError::InvalidFormat)?
            .try_into()
            .map_err(|_| FingerprintError::InvalidFormat)?;
        let tail_hash = hex::decode(parts[2])
            .map_err(|_| FingerprintError::InvalidFormat)?
            .try_into()
            .map_err(|_| FingerprintError::InvalidFormat)?;
        let file_size = parts[3]
            .parse()
            .map_err(|_| FingerprintError::InvalidFormat)?;

        Ok(Self {
            format,
            header_hash,
            tail_hash,
            file_size,
            tensor_count: None,
            metadata_version: None,
        })
    }

    /// Get a short ID suitable for display (first 8 chars of header hash)
    pub fn short_id(&self) -> String {
        hex::encode(&self.header_hash[..4])
    }
}

/// Compute MD5 hash of data
fn compute_md5(data: &[u8]) -> [u8; 16] {
    let mut hasher = Md5::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Detect model format from header bytes
fn detect_format(header: &[u8]) -> String {
    if header.len() < 4 {
        return "unknown".to_string();
    }

    // Check magic bytes
    match &header[..4] {
        // TFLite FlatBuffer magic or version
        [0x18, 0x00, 0x00, 0x00] | [0x1C, 0x00, 0x00, 0x00] | [0x20, 0x00, 0x00, 0x00] => {
            "tflite".to_string()
        }
        // GGUF magic: "GGUF" (0x46554747 LE)
        [0x47, 0x47, 0x55, 0x46] => "gguf".to_string(),
        // ONNX protobuf (starts with field tag for ir_version or model)
        [0x08, ..] | [0x12, ..] => {
            // Check for ONNX-specific patterns
            if header.len() > 10 && header[4..8] == [0x12, 0x04, 0x6f, 0x6e] {
                "onnx".to_string()
            } else {
                // Could be ONNX or other protobuf
                "onnx".to_string()
            }
        }
        // SafeTensors JSON header (starts with length + '{')
        _ if header.len() > 8 => {
            // SafeTensors: 8-byte length prefix + JSON starting with '{'
            let len = u64::from_le_bytes(header[..8].try_into().unwrap_or([0; 8]));
            if len > 0 && len < 1_000_000 && header.get(8) == Some(&b'{') {
                return "safetensors".to_string();
            }
            // CoreML mlmodel (protobuf)
            if header.starts_with(b"\x08\x01\x12") {
                return "coreml".to_string();
            }
            "unknown".to_string()
        }
        _ => "unknown".to_string(),
    }
}

/// Known model version entry
#[derive(Debug, Clone)]
pub struct ModelVersion {
    /// Full SHA256 hash (computed lazily when needed)
    pub full_hash: Option<[u8; 32]>,
    /// Human-readable version string
    pub version: String,
    /// Available patches for this version
    pub patches_available: Vec<String>,
}

/// Database of known model fingerprints
///
/// Used for quick lookup of model versions without computing
/// full hashes.
#[derive(Debug, Clone, Default)]
pub struct FingerprintDB {
    entries: HashMap<ModelFingerprint, ModelVersion>,
}

impl FingerprintDB {
    /// Create a new empty database
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add a fingerprint with version info
    pub fn add(&mut self, fingerprint: ModelFingerprint, version: ModelVersion) {
        self.entries.insert(fingerprint, version);
    }

    /// Look up a fingerprint
    pub fn lookup(&self, fingerprint: &ModelFingerprint) -> Option<&ModelVersion> {
        self.entries.get(fingerprint)
    }

    /// Find matching fingerprint (may match on subset of fields)
    pub fn find_matching(
        &self,
        fingerprint: &ModelFingerprint,
    ) -> Option<(&ModelFingerprint, &ModelVersion)> {
        self.entries.iter().find(|(fp, _)| fp.matches(fingerprint))
    }

    /// Get all entries
    pub fn entries(&self) -> impl Iterator<Item = (&ModelFingerprint, &ModelVersion)> {
        self.entries.iter()
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize database to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let entries: Vec<_> = self
            .entries
            .iter()
            .map(|(fp, ver)| {
                serde_json::json!({
                    "fingerprint": fp.to_compact_string(),
                    "version": ver.version,
                    "patches": ver.patches_available,
                })
            })
            .collect();

        serde_json::to_string_pretty(&entries)
    }

    /// Load database from JSON
    pub fn from_json(json: &str) -> Result<Self, FingerprintError> {
        let entries: Vec<serde_json::Value> =
            serde_json::from_str(json).map_err(|_| FingerprintError::InvalidFormat)?;

        let mut db = Self::new();

        for entry in entries {
            let fp_str = entry["fingerprint"]
                .as_str()
                .ok_or(FingerprintError::InvalidFormat)?;
            let fingerprint = ModelFingerprint::from_compact_string(fp_str)?;

            let version = ModelVersion {
                full_hash: None,
                version: entry["version"].as_str().unwrap_or("unknown").to_string(),
                patches_available: entry["patches"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
            };

            db.add(fingerprint, version);
        }

        Ok(db)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_fingerprint_from_bytes() {
        // Create test data (simulated TFLite header)
        let mut data = vec![0x18, 0x00, 0x00, 0x00]; // TFLite-like header
        data.extend(vec![0u8; 100_000]); // Padding to ensure we use header/tail

        let fp = ModelFingerprint::from_bytes(&data).unwrap();

        assert_eq!(fp.format, "tflite");
        assert_eq!(fp.file_size, data.len() as u64);
        assert!(!fp.short_id().is_empty());
    }

    #[test]
    fn test_fingerprint_small_file() {
        let data = vec![0x47, 0x47, 0x55, 0x46, 0x01, 0x02, 0x03, 0x04]; // Small GGUF-like

        let fp = ModelFingerprint::from_bytes(&data).unwrap();

        assert_eq!(fp.format, "gguf");
        assert_eq!(fp.file_size, 8);
        // For small files, header and tail hash should be the same
        assert_eq!(fp.header_hash, fp.tail_hash);
    }

    #[test]
    fn test_fingerprint_matching() {
        // Use 200KB file to ensure header (64KB) and tail (4KB) don't overlap
        let data1 = vec![0u8; 200_000];
        let data2 = vec![0u8; 200_000];
        let mut data3 = vec![0u8; 200_000];
        // Modify position 100,000 which is:
        // - After header (64KB = 65,536)
        // - Before tail starts (200,000 - 4,096 = 195,904)
        data3[100_000] = 1;

        let fp1 = ModelFingerprint::from_bytes(&data1).unwrap();
        let fp2 = ModelFingerprint::from_bytes(&data2).unwrap();
        let fp3 = ModelFingerprint::from_bytes(&data3).unwrap();

        assert!(fp1.matches(&fp2));
        assert!(fp1.matches(&fp3)); // Middle changes don't affect fingerprint
    }

    #[test]
    fn test_fingerprint_different_tail() {
        let data1 = vec![0u8; 100_000];
        let mut data2 = vec![0u8; 100_000];
        data2[99_999] = 1; // Different tail

        let fp1 = ModelFingerprint::from_bytes(&data1).unwrap();
        let fp2 = ModelFingerprint::from_bytes(&data2).unwrap();

        assert!(!fp1.matches(&fp2)); // Tail changes should affect fingerprint
    }

    #[test]
    fn test_fingerprint_compact_string_roundtrip() {
        let data = vec![0u8; 100_000];
        let fp = ModelFingerprint::from_bytes(&data).unwrap();

        let compact = fp.to_compact_string();
        let restored = ModelFingerprint::from_compact_string(&compact).unwrap();

        assert!(fp.matches(&restored));
        assert_eq!(fp.format, restored.format);
        assert_eq!(fp.file_size, restored.file_size);
    }

    #[test]
    fn test_fingerprint_from_reader() {
        let data = vec![0u8; 100_000];
        let cursor = Cursor::new(&data);

        let fp = ModelFingerprint::from_reader(cursor, data.len() as u64).unwrap();
        let fp_bytes = ModelFingerprint::from_bytes(&data).unwrap();

        assert!(fp.matches(&fp_bytes));
    }

    #[test]
    fn test_fingerprint_db() {
        let mut db = FingerprintDB::new();

        let fp = ModelFingerprint::from_bytes(&vec![0u8; 100_000]).unwrap();
        let version = ModelVersion {
            full_hash: None,
            version: "v1.0.0".to_string(),
            patches_available: vec!["patch1".to_string()],
        };

        db.add(fp.clone(), version);

        assert_eq!(db.len(), 1);
        assert!(db.lookup(&fp).is_some());
        assert_eq!(db.lookup(&fp).unwrap().version, "v1.0.0");
    }

    #[test]
    fn test_fingerprint_db_json_roundtrip() {
        let mut db = FingerprintDB::new();

        let fp = ModelFingerprint::from_bytes(&vec![0u8; 100_000]).unwrap();
        db.add(
            fp.clone(),
            ModelVersion {
                full_hash: None,
                version: "v1.0.0".to_string(),
                patches_available: vec!["patch1".to_string(), "patch2".to_string()],
            },
        );

        let json = db.to_json().unwrap();
        let restored = FingerprintDB::from_json(&json).unwrap();

        assert_eq!(restored.len(), 1);
        let (restored_fp, restored_ver) = restored.find_matching(&fp).unwrap();
        assert!(restored_fp.matches(&fp));
        assert_eq!(restored_ver.version, "v1.0.0");
        assert_eq!(restored_ver.patches_available.len(), 2);
    }

    #[test]
    fn test_format_detection() {
        // TFLite
        let tflite = vec![0x18, 0x00, 0x00, 0x00];
        assert_eq!(detect_format(&tflite), "tflite");

        // GGUF
        let gguf = vec![0x47, 0x47, 0x55, 0x46];
        assert_eq!(detect_format(&gguf), "gguf");

        // SafeTensors (8-byte length + JSON)
        let mut safetensors = vec![0; 8];
        safetensors[0] = 100; // length = 100
        safetensors.push(b'{');
        assert_eq!(detect_format(&safetensors), "safetensors");

        // Unknown
        let unknown = vec![0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(detect_format(&unknown), "unknown");
    }

    #[test]
    fn test_fingerprint_with_metadata() {
        let fp = ModelFingerprint::from_bytes(&vec![0u8; 100_000])
            .unwrap()
            .with_tensor_count(42)
            .with_version("v1.2.3".to_string());

        assert_eq!(fp.tensor_count, Some(42));
        assert_eq!(fp.metadata_version, Some("v1.2.3".to_string()));
    }
}
