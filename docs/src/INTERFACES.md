# Mallorn Interfaces

Trait definitions and API contracts for edge model delta updates.

## Core Traits

### ModelFormat

Abstraction over different model file formats.

```rust
/// A parseable model format (TFLite, GGUF, ONNX, etc.)
pub trait ModelFormat: Send + Sync {
    /// Format identifier
    fn format_id(&self) -> &'static str;
    
    /// File extensions this format handles
    fn extensions(&self) -> &[&'static str];
    
    /// Parse a model file into structured representation
    fn parse(&self, data: &[u8]) -> Result<ParsedModel, ParseError>;
    
    /// Serialize a parsed model back to bytes
    fn serialize(&self, model: &ParsedModel) -> Result<Vec<u8>, SerializeError>;
    
    /// Extract tensor metadata without full parse
    fn extract_tensor_info(&self, data: &[u8]) -> Result<Vec<TensorInfo>, ParseError>;
}

/// Registration for format auto-detection
pub struct FormatRegistry {
    formats: Vec<Box<dyn ModelFormat>>,
}

impl FormatRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, format: Box<dyn ModelFormat>);
    pub fn detect(&self, data: &[u8]) -> Option<&dyn ModelFormat>;
    pub fn by_extension(&self, ext: &str) -> Option<&dyn ModelFormat>;
}
```

### Differ

Computes deltas between model versions.

```rust
/// Computes differences between two models
pub trait Differ: Send + Sync {
    /// Compute a patch that transforms `old` into `new`
    fn diff(&self, old: &ParsedModel, new: &ParsedModel) -> Result<Patch, DiffError>;
    
    /// Compute diff with custom options
    fn diff_with_options(
        &self,
        old: &ParsedModel,
        new: &ParsedModel,
        options: &DiffOptions,
    ) -> Result<Patch, DiffError>;
}

/// Options for diff computation
#[derive(Debug, Clone)]
pub struct DiffOptions {
    /// Compression method for delta data
    pub compression: CompressionMethod,
    
    /// Minimum tensor size to diff (smaller tensors stored whole)
    pub min_tensor_size: usize,
    
    /// Enable neural-aware compression (ZipNN-style)
    pub neural_compression: bool,
    
    /// Target patch size (best-effort)
    pub target_size_hint: Option<usize>,
}

impl Default for DiffOptions {
    fn default() -> Self {
        Self {
            compression: CompressionMethod::Zstd { level: 3 },
            min_tensor_size: 1024,
            neural_compression: true,
            target_size_hint: None,
        }
    }
}
```

### Patcher

Applies patches to models.

```rust
/// Applies patches to transform models
pub trait Patcher: Send + Sync {
    /// Apply a patch to transform `old` into `new`
    fn patch(&self, old: &[u8], patch: &Patch) -> Result<Vec<u8>, PatchError>;
    
    /// Verify a patch can be applied (without applying)
    fn verify(&self, old: &[u8], patch: &Patch) -> Result<PatchVerification, PatchError>;
}

/// Streaming patcher for memory-constrained devices
pub trait StreamingPatcher {
    /// Initialize streaming patch application
    fn init(&mut self, old_reader: impl Read, patch_reader: impl Read) -> Result<(), PatchError>;
    
    /// Process one chunk, write output
    fn step(&mut self, output: &mut impl Write) -> Result<StepResult, PatchError>;
    
    /// Finalize and verify
    fn finalize(&mut self) -> Result<PatchVerification, PatchError>;
    
    /// Abort and cleanup
    fn abort(&mut self);
    
    /// Required buffer size
    fn buffer_size(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    Continue,
    Complete,
}
```

### Compressor

Neural-aware compression.

```rust
/// Compresses data with neural network awareness
pub trait Compressor: Send + Sync {
    /// Compress data
    fn compress(&self, data: &[u8], dtype: DataType) -> Result<Vec<u8>, CompressionError>;
    
    /// Decompress data
    fn decompress(&self, data: &[u8], dtype: DataType) -> Result<Vec<u8>, CompressionError>;
    
    /// Compression method identifier
    fn method(&self) -> CompressionMethod;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMethod {
    None,
    Zstd { level: i32 },
    Lz4,
    Neural { variant: NeuralCompressionVariant },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuralCompressionVariant {
    /// ZipNN-style exponent/mantissa separation
    ExponentGrouping,
    /// Byte-plane separation
    BytePlane,
}
```

## Data Types

### ParsedModel

```rust
/// A parsed model representation
#[derive(Debug, Clone)]
pub struct ParsedModel {
    /// Original format
    pub format: String,
    
    /// Model metadata
    pub metadata: ModelMetadata,
    
    /// Tensors in the model
    pub tensors: Vec<Tensor>,
    
    /// Operator graph (optional)
    pub graph: Option<OperatorGraph>,
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name/description
    pub name: Option<String>,
    
    /// Version info
    pub version: Option<String>,
    
    /// Custom metadata key-value pairs
    pub custom: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor name/identifier
    pub name: String,
    
    /// Shape dimensions
    pub shape: Vec<usize>,
    
    /// Data type
    pub dtype: DataType,
    
    /// Raw tensor data
    pub data: Vec<u8>,
    
    /// Quantization parameters (if quantized)
    pub quantization: Option<QuantizationParams>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
    Int8,
    UInt8,
    Int16,
    Int32,
    Int64,
    // Quantized types
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
}

#[derive(Debug, Clone)]
pub struct QuantizationParams {
    pub scale: f32,
    pub zero_point: i32,
}
```

### Patch

```rust
/// A model patch
#[derive(Debug, Clone)]
pub struct Patch {
    /// Patch format version
    pub version: u32,
    
    /// Source model hash
    pub source_hash: [u8; 32],
    
    /// Target model hash
    pub target_hash: [u8; 32],
    
    /// Patch operations
    pub operations: Vec<PatchOperation>,
    
    /// Compression used
    pub compression: CompressionMethod,
    
    /// Patch metadata
    pub metadata: PatchMetadata,
}

#[derive(Debug, Clone)]
pub enum PatchOperation {
    /// Replace tensor entirely
    ReplaceTensor {
        name: String,
        data: Vec<u8>,
    },
    
    /// Delta update to tensor
    DeltaTensor {
        name: String,
        delta: Vec<u8>,
        delta_format: DeltaFormat,
    },
    
    /// Update metadata only
    UpdateMetadata {
        key: String,
        value: String,
    },
    
    /// Copy tensor unchanged (for reference)
    CopyTensor {
        name: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaFormat {
    /// XOR delta
    Xor,
    /// BSDiff-style
    BsDiff,
    /// Custom tensor-aware
    TensorAware,
}

#[derive(Debug, Clone)]
pub struct PatchMetadata {
    /// Source model version
    pub source_version: Option<String>,
    
    /// Target model version
    pub target_version: Option<String>,
    
    /// Creation timestamp
    pub created_at: u64,
    
    /// Patch description
    pub description: Option<String>,
}
```

### TensorInfo

```rust
/// Lightweight tensor metadata (no data)
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub offset: usize,
    pub size: usize,
}
```

### Verification

```rust
/// Result of patch verification
#[derive(Debug, Clone)]
pub struct PatchVerification {
    /// Source hash matches?
    pub source_valid: bool,
    
    /// Patch integrity valid?
    pub patch_valid: bool,
    
    /// Expected target hash
    pub expected_target: [u8; 32],
    
    /// Actual target hash (after apply)
    pub actual_target: Option<[u8; 32]>,
    
    /// Patch statistics
    pub stats: PatchStats,
}

#[derive(Debug, Clone)]
pub struct PatchStats {
    /// Original model size
    pub source_size: usize,
    
    /// Target model size
    pub target_size: usize,
    
    /// Patch size
    pub patch_size: usize,
    
    /// Compression ratio (target_size / patch_size)
    pub compression_ratio: f64,
    
    /// Number of tensors modified
    pub tensors_modified: usize,
    
    /// Number of tensors unchanged
    pub tensors_unchanged: usize,
}
```

## Public API

### Rust API

```rust
// High-level convenience functions

/// Create a patch between two model files
pub fn create_patch(
    old_path: impl AsRef<Path>,
    new_path: impl AsRef<Path>,
    options: Option<DiffOptions>,
) -> Result<Patch, MallornError>;

/// Apply a patch to a model file
pub fn apply_patch(
    old_path: impl AsRef<Path>,
    patch_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<PatchVerification, MallornError>;

/// Verify a patch without applying
pub fn verify_patch(
    old_path: impl AsRef<Path>,
    patch_path: impl AsRef<Path>,
) -> Result<PatchVerification, MallornError>;

/// Get patch information
pub fn patch_info(patch_path: impl AsRef<Path>) -> Result<PatchInfo, MallornError>;

// Builder pattern for advanced usage

pub struct MallornBuilder {
    formats: FormatRegistry,
    compressor: Box<dyn Compressor>,
    options: DiffOptions,
}

impl MallornBuilder {
    pub fn new() -> Self;
    pub fn with_format(self, format: impl ModelFormat + 'static) -> Self;
    pub fn with_compressor(self, compressor: impl Compressor + 'static) -> Self;
    pub fn with_options(self, options: DiffOptions) -> Self;
    pub fn build(self) -> Mallorn;
}

pub struct Mallorn {
    // ...
}

impl Mallorn {
    pub fn diff(&self, old: &[u8], new: &[u8]) -> Result<Patch, MallornError>;
    pub fn patch(&self, old: &[u8], patch: &Patch) -> Result<Vec<u8>, MallornError>;
    pub fn verify(&self, old: &[u8], patch: &Patch) -> Result<PatchVerification, MallornError>;
}
```

### Python API (PyO3)

```python
import mallorn

# Simple API
patch = mallorn.create_patch("model_v1.tflite", "model_v2.tflite")
patch.save("update.tflp")

result = mallorn.apply_patch("model_v1.tflite", "update.tflp", "model_v2.tflite")
print(f"Compression ratio: {result.compression_ratio:.1f}x")

# Verification
verification = mallorn.verify_patch("model_v1.tflite", "update.tflp")
assert verification.source_valid
assert verification.patch_valid

# Patch info
info = mallorn.patch_info("update.tflp")
print(f"Patch size: {info.patch_size} bytes")
print(f"Target hash: {info.target_hash.hex()}")

# Advanced options
options = mallorn.DiffOptions(
    compression="zstd",
    compression_level=6,
    neural_compression=True,
)
patch = mallorn.create_patch("old.gguf", "new.gguf", options=options)
```

### CLI

```bash
# Create a patch
mallorn diff model_v1.tflite model_v2.tflite -o update.tflp

# With options
mallorn diff old.gguf new.gguf -o update.ggup \
    --compression zstd \
    --level 6 \
    --neural

# Apply a patch
mallorn patch model_v1.tflite update.tflp -o model_v2.tflite

# Verify a patch
mallorn verify model_v1.tflite update.tflp

# Get patch info
mallorn info update.tflp

# Output:
# Patch: update.tflp
# Format: TFLite Patch v1
# Source: sha256:abc123...
# Target: sha256:def456...
# Size: 2.3 MB (5.2% of target)
# Tensors: 42 modified, 8 unchanged
# Compression: zstd (level 6) + neural
```

### C API (mallorn-lite)

```c
#include "mallorn_lite.h"

// Types
typedef struct mallorn_patcher mallorn_patcher_t;

typedef enum {
    MALLORN_OK = 0,
    MALLORN_CONTINUE = 1,
    MALLORN_ERROR_INVALID_PATCH = -1,
    MALLORN_ERROR_HASH_MISMATCH = -2,
    MALLORN_ERROR_BUFFER_TOO_SMALL = -3,
    MALLORN_ERROR_IO = -4,
} mallorn_result_t;

// Initialize patcher with working buffer
mallorn_result_t mallorn_init(
    mallorn_patcher_t* patcher,
    uint8_t* buffer,
    size_t buffer_size
);

// Set source model reader
mallorn_result_t mallorn_set_source(
    mallorn_patcher_t* patcher,
    mallorn_read_fn read_fn,
    void* read_ctx
);

// Set patch reader
mallorn_result_t mallorn_set_patch(
    mallorn_patcher_t* patcher,
    mallorn_read_fn read_fn,
    void* read_ctx
);

// Set output writer
mallorn_result_t mallorn_set_output(
    mallorn_patcher_t* patcher,
    mallorn_write_fn write_fn,
    void* write_ctx
);

// Process one chunk (call in loop)
mallorn_result_t mallorn_step(mallorn_patcher_t* patcher);

// Verify final hash
mallorn_result_t mallorn_verify(
    mallorn_patcher_t* patcher,
    uint8_t target_hash[32]
);

// Abort and cleanup
void mallorn_abort(mallorn_patcher_t* patcher);

// Get required buffer size for patch
size_t mallorn_required_buffer(const uint8_t* patch_header, size_t header_size);
```

## Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum MallornError {
    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),
    
    #[error("Diff error: {0}")]
    Diff(#[from] DiffError),
    
    #[error("Patch error: {0}")]
    Patch(#[from] PatchError),
    
    #[error("Compression error: {0}")]
    Compression(#[from] CompressionError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Unknown format: {0}")]
    UnknownFormat(String),
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Invalid magic bytes")]
    InvalidMagic,
    
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    
    #[error("Malformed data: {0}")]
    Malformed(String),
}

#[derive(Debug, thiserror::Error)]
pub enum PatchError {
    #[error("Source hash mismatch")]
    SourceHashMismatch,
    
    #[error("Target hash mismatch")]
    TargetHashMismatch,
    
    #[error("Patch corrupted")]
    Corrupted,
    
    #[error("Incompatible version")]
    IncompatibleVersion,
}
```

## File Formats

### TFLite Patch (.tflp)

```
┌────────────────────────────────────────┐
│ Magic: "TFLP" (4 bytes)                │
├────────────────────────────────────────┤
│ Version: u32                           │
├────────────────────────────────────────┤
│ Source Hash: [u8; 32]                  │
├────────────────────────────────────────┤
│ Target Hash: [u8; 32]                  │
├────────────────────────────────────────┤
│ Metadata Length: u32                   │
├────────────────────────────────────────┤
│ Metadata: JSON (compressed)            │
├────────────────────────────────────────┤
│ Num Operations: u32                    │
├────────────────────────────────────────┤
│ Operations: [PatchOp, ...]             │
├────────────────────────────────────────┤
│ Checksum: u32 (CRC32)                  │
└────────────────────────────────────────┘
```

### GGUF Patch (.ggup)

```
┌────────────────────────────────────────┐
│ Magic: "GGUP" (4 bytes)                │
├────────────────────────────────────────┤
│ Version: u32                           │
├────────────────────────────────────────┤
│ Source Hash: [u8; 32]                  │
├────────────────────────────────────────┤
│ Target Hash: [u8; 32]                  │
├────────────────────────────────────────┤
│ Metadata Patches: [MetaPatch, ...]     │
├────────────────────────────────────────┤
│ Tensor Patches: [TensorPatch, ...]     │
├────────────────────────────────────────┤
│ Checksum: u32 (CRC32)                  │
└────────────────────────────────────────┘
```
