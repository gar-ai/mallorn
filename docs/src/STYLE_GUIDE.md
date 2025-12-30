# Mallorn Style Guide

Coding conventions for edge model delta updates.

## Rust Conventions

### Project Structure

```
crate-name/
├── src/
│   ├── lib.rs              # Public API, re-exports
│   ├── types.rs            # Shared types
│   ├── error.rs            # Error types
│   ├── feature/
│   │   ├── mod.rs          # Module exports
│   │   └── impl.rs         # Implementation
│   └── internal/           # Private helpers
│       └── mod.rs
├── tests/
│   ├── invariants/         # Must-pass tests
│   └── hypotheses/         # Performance tests
├── benches/
└── examples/
```

### Naming Conventions

```rust
// Types: PascalCase
pub struct TensorDelta { ... }
pub enum CompressionMethod { ... }
pub trait ModelFormat { ... }

// Functions/methods: snake_case
pub fn create_patch(...) -> Result<Patch, Error> { ... }
fn parse_header(data: &[u8]) -> Result<Header, ParseError> { ... }

// Constants: SCREAMING_SNAKE_CASE
pub const MAX_TENSOR_SIZE: usize = 1 << 30;  // 1GB
const MAGIC_BYTES: &[u8] = b"TFLP";

// Modules: snake_case
mod tensor_differ;
mod compression;
```

### Error Handling

```rust
// Use thiserror for error types
#[derive(Debug, thiserror::Error)]
pub enum MallornError {
    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Source hash mismatch: expected {expected}, got {actual}")]
    SourceHashMismatch {
        expected: String,
        actual: String,
    },
}

// Use Result everywhere, avoid unwrap in library code
pub fn parse(data: &[u8]) -> Result<Model, ParseError> {
    let header = parse_header(data)?;
    let tensors = parse_tensors(data, &header)?;
    Ok(Model { header, tensors })
}

// In tests, unwrap is fine
#[test]
fn test_parse() {
    let model = parse(TEST_DATA).unwrap();
    assert_eq!(model.tensors.len(), 10);
}
```

### Documentation

```rust
/// Creates a patch that transforms `old` into `new`.
///
/// # Arguments
///
/// * `old` - Source model bytes
/// * `new` - Target model bytes
/// * `options` - Diff options (compression, etc.)
///
/// # Returns
///
/// A `Patch` that can be applied to `old` to produce `new`.
///
/// # Errors
///
/// Returns `DiffError::UnsupportedFormat` if the model format
/// is not recognized.
///
/// # Example
///
/// ```rust
/// use mallorn::{create_patch, DiffOptions};
///
/// let patch = create_patch(&old_model, &new_model, None)?;
/// assert!(patch.operations.len() > 0);
/// ```
pub fn create_patch(
    old: &[u8],
    new: &[u8],
    options: Option<DiffOptions>,
) -> Result<Patch, DiffError> {
    // ...
}
```

### Traits

```rust
// Traits should be object-safe when possible
pub trait Compressor: Send + Sync {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError>;
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError>;
}

// Use associated types for complex generics
pub trait ModelFormat {
    type Error: std::error::Error;
    
    fn parse(&self, data: &[u8]) -> Result<ParsedModel, Self::Error>;
}
```

### Performance

```rust
// Prefer borrowing over cloning
fn process(data: &[u8]) -> Result<(), Error>  // Good
fn process(data: Vec<u8>) -> Result<(), Error> // Avoid

// Use iterators
tensors.iter()
    .filter(|t| t.size > threshold)
    .map(|t| compress(t))
    .collect()

// Avoid allocations in hot paths
// Bad: allocates on every call
fn hot_path(data: &[u8]) -> Vec<u8> {
    let mut buffer = Vec::new();  // Allocation!
    // ...
}

// Good: reuse buffer
fn hot_path(data: &[u8], buffer: &mut Vec<u8>) {
    buffer.clear();
    // ...
}
```

### Unsafe Code

```rust
// Minimize unsafe, document invariants
/// # Safety
///
/// `ptr` must be valid for reads of `len` bytes.
/// The memory must be properly aligned for `T`.
unsafe fn read_tensor<T>(ptr: *const u8, len: usize) -> &[T] {
    // ...
}

// Prefer safe abstractions
// Instead of raw pointer manipulation, use byteorder crate
use byteorder::{LittleEndian, ReadBytesExt};
let value = cursor.read_u32::<LittleEndian>()?;
```

## Testing Conventions

### Test Organization

```rust
// Unit tests in same file
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic() { ... }
}

// Integration tests in tests/
// tests/invariants/roundtrip.rs
#[test]
fn invariant_roundtrip_lossless() { ... }

// tests/hypotheses/compression.rs
#[test]
fn hypothesis_compression_ratio() { ... }
```

### Test Naming

```rust
// Invariants: invariant_<what_must_be_true>
#[test]
fn invariant_roundtrip_produces_identical_output() { ... }

#[test]
fn invariant_corruption_is_detected() { ... }

// Hypotheses: hypothesis_<expected_behavior>
#[test]
fn hypothesis_patch_smaller_than_10_percent() { ... }

#[test]
fn hypothesis_apply_faster_than_100ms() { ... }

// Exploration: exploration_<what_measuring>
#[test]
#[ignore]
fn exploration_compression_ratios_by_model_type() { ... }
```

### Assertions

```rust
// Use descriptive assertions
assert!(
    ratio < 0.10,
    "Expected patch ratio <10%, got {:.1}%",
    ratio * 100.0
);

// Use assert_eq! for equality
assert_eq!(output, expected, "Roundtrip must be lossless");

// Use proptest for property testing
proptest! {
    #[test]
    fn roundtrip_any_data(data: Vec<u8>) {
        let compressed = compress(&data);
        let decompressed = decompress(&compressed);
        assert_eq!(data, decompressed);
    }
}
```

## Embedded C Conventions

### mallorn-lite Style

```c
// All functions prefixed with mallorn_
mallorn_result_t mallorn_init(...);
mallorn_result_t mallorn_step(...);

// Types suffixed with _t
typedef struct mallorn_patcher mallorn_patcher_t;
typedef enum mallorn_result mallorn_result_t;

// Constants prefixed with MALLORN_
#define MALLORN_OK 0
#define MALLORN_ERROR -1
#define MALLORN_MAX_BUFFER 1024

// No dynamic allocation in hot path
// All buffers passed by caller
mallorn_result_t mallorn_init(
    mallorn_patcher_t* patcher,
    uint8_t* buffer,        // Caller-provided
    size_t buffer_size      // Caller knows size
);
```

### Memory Safety

```c
// Bounds checking on all array access
if (offset + size > buffer_size) {
    return MALLORN_ERROR_BUFFER_TOO_SMALL;
}

// No malloc/free in streaming path
// All memory is caller-managed

// Clear sensitive data
memset(buffer, 0, buffer_size);
```

## Git Conventions

### Commit Messages

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `refactor`: Code change that neither fixes nor adds
- `test`: Adding tests
- `docs`: Documentation
- `chore`: Maintenance

Examples:
```
feat(tflite): add tensor-aware diffing

Implement alignment on tensor boundaries for improved
compression ratios.

Closes #42
```

```
perf(compression): batch exponent grouping

Process 256 bytes at a time instead of one-by-one.
Improves compression speed by 3x.
```

### Branches

```
main              # Stable, release-ready
dev               # Integration branch
feat/<name>       # Feature branches
fix/<issue>       # Bug fix branches
```

## Formatting

### Rust

```bash
# Format before commit
cargo fmt

# Check in CI
cargo fmt --check
```

### C

```bash
# Use clang-format with project .clang-format
clang-format -i mallorn_lite.c mallorn_lite.h
```

### Markdown

- One sentence per line (for better diffs)
- Use fenced code blocks with language
- Use ASCII diagrams for architecture

## CI Requirements

Every PR must pass:
- `cargo fmt --check`
- `cargo clippy -- -D warnings`
- `cargo test` (invariants)
- `cargo test --test hypotheses` (warnings OK)
- `cargo doc` (no warnings)
