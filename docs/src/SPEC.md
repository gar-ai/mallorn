# Mallorn

> *"Their bark was silver and smooth, and their boughs somewhat upswept after the manner of the beech; but they never grew save in the land of Lórien."*

Edge model delta updates for resource-constrained devices. Reduce OTA bandwidth from full model transfers to minimal patches.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MALLORN                                     │
│            Edge Model Delta Updates                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Cloud/Server                              Edge Device              │
│   ┌──────────────┐                         ┌──────────────┐         │
│   │  Model v1.0  │                         │  Model v1.0  │         │
│   │    (2GB)     │                         │   (Flash)    │         │
│   └──────┬───────┘                         └──────────────┘         │
│          │                                        ▲                  │
│          ▼                                        │                  │
│   ┌──────────────┐      OTA Update         ┌─────┴────────┐         │
│   │  Model v1.1  │  ─────────────────────▶ │ Delta Patch  │         │
│   │    (2GB)     │       (50MB)            │   Applied    │         │
│   └──────────────┘                         └──────────────┘         │
│          │                                        │                  │
│          ▼                                        ▼                  │
│   ┌──────────────┐                         ┌──────────────┐         │
│   │ Delta Engine │                         │  Model v1.1  │         │
│   │  - Diff      │                         │   (Flash)    │         │
│   │  - Compress  │                         └──────────────┘         │
│   │  - Package   │                                                   │
│   └──────────────┘                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## The Problem

| Scenario | Full Update | Delta Update | Savings |
|----------|-------------|--------------|---------|
| 200KB TFLite over LoRa | 6+ hours | 15 minutes | 96% |
| 2GB GGUF over cellular | $2-5 data cost | $0.10-0.25 | 95% |
| 500MB ONNX to fleet of 10K devices | 5PB transfer | 250TB | 95% |

Current approaches treat models as opaque binaries. We can do better by exploiting neural network structure.

## Target Formats

| Format | Use Case | Priority |
|--------|----------|----------|
| **TFLite** | Mobile/embedded ML | P0 (MVP) |
| **GGUF** | Local LLM inference (llama.cpp) | P0 (MVP) |
| **ONNX** | Cross-platform deployment | P1 |
| **TFLite Micro** | MCU deployment | P1 |
| **Core ML** | Apple devices | P2 |

## Target Constraints

| Constraint | Target | Notes |
|------------|--------|-------|
| Patch device RAM | 1KB minimum | HPatchLite compatibility |
| Patch size | <5% of model size | For minor updates |
| Apply time | <10s for 100MB model | On mid-tier mobile |
| Atomic updates | Required | A/B slots or journaling |
| Rollback | Required | Corrupted update recovery |

## Architecture

```
mallorn/
├── Cargo.toml
├── crates/
│   ├── mallorn-core/           # Shared primitives
│   │   ├── compression/         # ZipNN-style neural compression
│   │   ├── diff/                # Binary diff algorithms
│   │   └── format/              # Model format parsers
│   │
│   ├── mallorn-tflite/         # TFLite delta engine
│   │   ├── parser/              # FlatBuffer parsing
│   │   ├── diff/                # Tensor-aware diffing
│   │   └── patch/               # Patch generation/application
│   │
│   ├── mallorn-gguf/           # GGUF delta engine
│   │   ├── parser/              # GGUF format parsing
│   │   ├── diff/                # Quantized weight diffing
│   │   └── patch/               # Patch generation/application
│   │
│   ├── mallorn-onnx/           # ONNX delta engine (P1)
│   │
│   └── mallorn-cli/            # Command-line tools
│
├── embedded/
│   ├── mallorn-lite/           # Minimal C library for MCUs
│   └── examples/
│       ├── esp32/
│       ├── stm32/
│       └── nrf52/
│
└── python/
    └── mallorn/                # Python bindings
```

## Core Algorithms

### 1. Tensor-Aware Binary Diffing

Unlike generic binary diff (bsdiff, HDiffPatch), we exploit model structure:

```rust
/// Tensor-aware diff that aligns on layer boundaries
pub struct TensorAwareDiff {
    /// Minimum tensor size to diff separately (bytes)
    min_tensor_size: usize,
    /// Compression for delta data
    compressor: Box<dyn Compressor>,
}

impl TensorAwareDiff {
    /// Diff two models, producing tensor-aligned patches
    pub fn diff(&self, old: &Model, new: &Model) -> Result<DeltaPatch> {
        let mut patches = Vec::new();
        
        // Align tensors by name/index
        for (old_tensor, new_tensor) in align_tensors(old, new) {
            match (old_tensor, new_tensor) {
                // Tensor unchanged
                (Some(o), Some(n)) if o.data == n.data => {
                    // No patch needed
                }
                // Tensor modified
                (Some(o), Some(n)) => {
                    let delta = self.diff_tensor(o, n)?;
                    patches.push(TensorPatch::Modified {
                        name: n.name.clone(),
                        delta,
                    });
                }
                // Tensor added
                (None, Some(n)) => {
                    patches.push(TensorPatch::Added {
                        name: n.name.clone(),
                        data: self.compressor.compress(&n.data)?,
                    });
                }
                // Tensor removed
                (Some(o), None) => {
                    patches.push(TensorPatch::Removed {
                        name: o.name.clone(),
                    });
                }
                _ => {}
            }
        }
        
        Ok(DeltaPatch { patches, metadata: new.metadata.clone() })
    }
    
    /// Diff individual tensor using best strategy for dtype
    fn diff_tensor(&self, old: &Tensor, new: &Tensor) -> Result<Vec<u8>> {
        match old.dtype {
            // Float tensors: XOR + neural-aware compression
            DType::Float32 | DType::Float16 | DType::BFloat16 => {
                let xor_delta = xor_bytes(&old.data, &new.data);
                self.compressor.compress(&xor_delta)
            }
            // Quantized tensors: direct diff (already compressed)
            DType::Int8 | DType::Int4 => {
                let xor_delta = xor_bytes(&old.data, &new.data);
                // Less compressible, use fast LZ4
                lz4_compress(&xor_delta)
            }
        }
    }
}
```

### 2. ZipNN-Style Neural Compression

Exploit floating-point exponent skewness in neural network weights:

```rust
/// Neural-network-aware compression
/// Key insight: NN weights have skewed exponent distributions
pub struct NeuralCompressor {
    /// Compression level (1-22 for zstd backend)
    level: i32,
    /// Use exponent-aware coding
    use_exponent_coding: bool,
}

impl NeuralCompressor {
    /// Compress with knowledge of FP weight distribution
    pub fn compress(&self, data: &[u8], dtype: DType) -> Result<Vec<u8>> {
        match dtype {
            DType::Float32 => self.compress_f32(data),
            DType::Float16 | DType::BFloat16 => self.compress_f16(data),
            _ => self.compress_generic(data),
        }
    }
    
    fn compress_f32(&self, data: &[u8]) -> Result<Vec<u8>> {
        if !self.use_exponent_coding {
            return zstd::compress(data, self.level);
        }
        
        // Separate exponent and mantissa bytes
        // Float32: [sign:1][exp:8][mantissa:23]
        let floats: &[f32] = bytemuck::cast_slice(data);
        
        let mut exponents = Vec::with_capacity(floats.len());
        let mut mantissas = Vec::with_capacity(floats.len() * 3);
        
        for &f in floats {
            let bits = f.to_bits();
            let exp = ((bits >> 23) & 0xFF) as u8;
            let mantissa = bits & 0x7FFFFF;
            
            exponents.push(exp);
            mantissas.extend_from_slice(&mantissa.to_le_bytes()[..3]);
        }
        
        // Exponents compress extremely well (clustered distribution)
        let compressed_exp = zstd::compress(&exponents, self.level + 3)?;
        // Mantissas less so, use lower level
        let compressed_mant = zstd::compress(&mantissas, self.level)?;
        
        // Pack together
        let mut result = Vec::new();
        result.extend_from_slice(&(compressed_exp.len() as u32).to_le_bytes());
        result.extend_from_slice(&compressed_exp);
        result.extend_from_slice(&compressed_mant);
        
        Ok(result)
    }
}
```

### 3. Incremental Patch Application (1KB RAM)

Based on HDiffPatch's HPatchLite for extreme memory constraints:

```rust
/// Streaming patch application for memory-constrained devices
/// Can operate with as little as 1KB working memory
pub struct StreamingPatcher<R: Read, W: Write> {
    /// Input: old model (read stream)
    old_reader: R,
    /// Output: new model (write stream)
    new_writer: W,
    /// Working buffer (configurable size)
    buffer: Vec<u8>,
}

impl<R: Read, W: Write> StreamingPatcher<R, W> {
    pub fn new(old: R, new: W, buffer_size: usize) -> Self {
        Self {
            old_reader: old,
            new_writer: new,
            buffer: vec![0u8; buffer_size],
        }
    }
    
    /// Apply patch with minimal memory
    /// Patch format: [op:1][len:var][data:len]
    pub fn apply<P: Read>(&mut self, patch: &mut P) -> Result<()> {
        loop {
            let op = match read_u8(patch) {
                Ok(op) => op,
                Err(_) => break, // End of patch
            };
            
            match op {
                OP_COPY_OLD => {
                    // Copy bytes from old model
                    let offset = read_varint(patch)?;
                    let len = read_varint(patch)?;
                    self.copy_from_old(offset, len)?;
                }
                OP_INSERT_NEW => {
                    // Insert new data from patch
                    let len = read_varint(patch)?;
                    self.insert_from_patch(patch, len)?;
                }
                OP_XOR_DELTA => {
                    // XOR delta against old data
                    let offset = read_varint(patch)?;
                    let len = read_varint(patch)?;
                    self.apply_xor_delta(patch, offset, len)?;
                }
                _ => return Err(PatchError::InvalidOp(op)),
            }
        }
        
        Ok(())
    }
    
    fn copy_from_old(&mut self, offset: u64, len: u64) -> Result<()> {
        // Seek and copy in buffer-sized chunks
        // Works with sequential flash reads
        todo!()
    }
}
```

### 4. A/B Slot Management

Atomic updates with rollback support:

```rust
/// A/B slot manager for atomic model updates
pub struct SlotManager {
    /// Slot A path/offset
    slot_a: SlotInfo,
    /// Slot B path/offset
    slot_b: SlotInfo,
    /// Current active slot
    active: Slot,
    /// Metadata storage (persistent)
    metadata_path: PathBuf,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Slot { A, B }

#[derive(Serialize, Deserialize)]
pub struct SlotInfo {
    pub path: PathBuf,
    pub version: String,
    pub checksum: [u8; 32],
    pub valid: bool,
}

impl SlotManager {
    /// Begin update to inactive slot
    pub fn begin_update(&mut self) -> Result<&mut SlotInfo> {
        let target = match self.active {
            Slot::A => &mut self.slot_b,
            Slot::B => &mut self.slot_a,
        };
        target.valid = false;
        self.persist_metadata()?;
        Ok(target)
    }
    
    /// Commit update: mark slot valid, switch active
    pub fn commit_update(&mut self, slot: Slot, version: &str, checksum: [u8; 32]) -> Result<()> {
        let target = match slot {
            Slot::A => &mut self.slot_a,
            Slot::B => &mut self.slot_b,
        };
        target.version = version.to_string();
        target.checksum = checksum;
        target.valid = true;
        self.active = slot;
        self.persist_metadata()?;
        Ok(())
    }
    
    /// Rollback to previous slot (if valid)
    pub fn rollback(&mut self) -> Result<()> {
        let previous = match self.active {
            Slot::A => Slot::B,
            Slot::B => Slot::A,
        };
        let prev_info = match previous {
            Slot::A => &self.slot_a,
            Slot::B => &self.slot_b,
        };
        if !prev_info.valid {
            return Err(SlotError::NoPreviousValid);
        }
        self.active = previous;
        self.persist_metadata()?;
        Ok(())
    }
}
```

## File Formats

### TFLite Patch Format (.tflp)

```
┌─────────────────────────────────────────┐
│           TFLITE PATCH FORMAT           │
├─────────────────────────────────────────┤
│ Magic: "TFLP" (4 bytes)                 │
│ Version: u16                            │
│ Flags: u16                              │
│   bit 0: has_metadata_patch             │
│   bit 1: has_tensor_patches             │
│   bit 2: neural_compression             │
├─────────────────────────────────────────┤
│ Source Model Hash: [u8; 32]             │
│ Target Model Hash: [u8; 32]             │
│ Target Model Version: string            │
├─────────────────────────────────────────┤
│ Metadata Patch (if flag set):           │
│   - FlatBuffer schema changes           │
│   - Operator additions/removals         │
├─────────────────────────────────────────┤
│ Tensor Patches:                         │
│   ┌─────────────────────────────────┐  │
│   │ Tensor Index: varint            │  │
│   │ Patch Type: u8                  │  │
│   │   0 = unchanged                 │  │
│   │   1 = replaced                  │  │
│   │   2 = xor_delta                 │  │
│   │   3 = removed                   │  │
│   │ Compressed Data Length: varint  │  │
│   │ Compressed Data: [u8]           │  │
│   └─────────────────────────────────┘  │
│   ... more tensor patches ...           │
├─────────────────────────────────────────┤
│ Footer:                                 │
│   - Patch checksum                      │
│   - Uncompressed target size            │
└─────────────────────────────────────────┘
```

### GGUF Patch Format (.ggup)

```
┌─────────────────────────────────────────┐
│            GGUF PATCH FORMAT            │
├─────────────────────────────────────────┤
│ Magic: "GGUP" (4 bytes)                 │
│ Version: u32                            │
│ Source GGUF Version: u32                │
│ Target GGUF Version: u32                │
├─────────────────────────────────────────┤
│ Metadata Patches:                       │
│   - Key-value additions                 │
│   - Key-value modifications             │
│   - Key-value removals                  │
├─────────────────────────────────────────┤
│ Tensor Patches:                         │
│   ┌─────────────────────────────────┐  │
│   │ Tensor Name: string             │  │
│   │ Quantization Type: u32          │  │
│   │ Patch Type: u8                  │  │
│   │ Delta Data: [u8]                │  │
│   └─────────────────────────────────┘  │
│   ... more tensor patches ...           │
├─────────────────────────────────────────┤
│ Footer Checksum                         │
└─────────────────────────────────────────┘
```

## Public API

### Rust API

```rust
// Generate delta patch
pub fn create_patch(old_model: &Path, new_model: &Path, output: &Path) -> Result<PatchStats>;

// Apply delta patch
pub fn apply_patch(model: &Path, patch: &Path, output: &Path) -> Result<()>;

// Verify patch applicability
pub fn verify_patch(model: &Path, patch: &Path) -> Result<VerifyResult>;

// Streaming API for embedded
pub fn create_streaming_patcher(buffer_size: usize) -> StreamingPatcher;
```

### Python API

```python
import mallorn

# Create patch
stats = mallorn.create_patch(
    old_model="model_v1.tflite",
    new_model="model_v2.tflite",
    output="v1_to_v2.tflp",
)
print(f"Patch size: {stats.patch_size} ({stats.compression_ratio:.1f}x smaller)")

# Apply patch
mallorn.apply_patch(
    model="model_v1.tflite",
    patch="v1_to_v2.tflp",
    output="model_v2.tflite",
)

# Verify before apply
result = mallorn.verify_patch("model_v1.tflite", "v1_to_v2.tflp")
if result.compatible:
    mallorn.apply_patch(...)
```

### CLI

```bash
# Create patch
mallorn diff model_v1.tflite model_v2.tflite -o update.tflp

# Apply patch
mallorn patch model_v1.tflite update.tflp -o model_v2.tflite

# Verify patch
mallorn verify model_v1.tflite update.tflp

# Info about patch
mallorn info update.tflp

# GGUF support
mallorn diff model_v1.gguf model_v2.gguf -o update.ggup
```

### C API (for embedded)

```c
// Minimal C API for MCU integration
#include "mallorn_lite.h"

// Initialize patcher with buffer
mallorn_patcher_t patcher;
uint8_t buffer[1024];  // 1KB working memory
mallorn_init(&patcher, buffer, sizeof(buffer));

// Open streams
mallorn_set_old_model(&patcher, flash_read_fn, old_model_offset);
mallorn_set_output(&patcher, flash_write_fn, new_model_offset);
mallorn_set_patch(&patcher, ota_read_fn, patch_data);

// Apply incrementally
while (mallorn_step(&patcher) == PALANTIR_CONTINUE) {
    // Can yield to other tasks here
    watchdog_feed();
}

if (mallorn_status(&patcher) == PALANTIR_SUCCESS) {
    // Verify checksum
    if (mallorn_verify(&patcher)) {
        slot_manager_commit();
    }
}
```

## Implementation Plan

### Phase 1: Core + TFLite (Week 1-4)

- [ ] TFLite FlatBuffer parser
- [ ] Tensor extraction and alignment
- [ ] XOR delta encoding
- [ ] Zstd compression
- [ ] Patch generation CLI
- [ ] Patch application CLI
- [ ] Python bindings

### Phase 2: GGUF Support (Week 5-6)

- [ ] GGUF format parser
- [ ] Quantized tensor diffing
- [ ] .ggup patch format
- [ ] llama.cpp integration testing

### Phase 3: Neural Compression (Week 7-8)

- [ ] ZipNN-style exponent coding
- [ ] Benchmarks vs generic compression
- [ ] Integration with patch formats

### Phase 4: Embedded Support (Week 9-12)

- [ ] mallorn-lite C library
- [ ] HPatchLite-style streaming
- [ ] A/B slot manager
- [ ] ESP32 example
- [ ] STM32 example
- [ ] nRF52 example

### Phase 5: ONNX + Polish (Week 13+)

- [ ] ONNX format support
- [ ] Documentation
- [ ] Performance optimization
- [ ] Production hardening

## Benchmarks

### Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Patch size (minor update) | <5% of model | Similar architecture, weight updates only |
| Patch size (major update) | <30% of model | Layer additions/removals |
| Compression ratio | 10-50x vs naive | Depends on change magnitude |
| Patch generation | >100 MB/s | Server-side |
| Patch application | >50 MB/s | Mobile device |
| Minimum RAM | 1KB | Streaming mode |
| Checksum verification | Required | SHA-256 or xxHash |

### Comparison Baselines

| Baseline | What It Measures |
|----------|------------------|
| Full model transfer | Worst case |
| bsdiff/bspatch | Generic binary diff |
| HDiffPatch | Fast binary diff |
| ZipNN alone | Compression without diffing |

## Research References

### Delta Updates

| Paper | Key Insight |
|-------|-------------|
| **ImPart** (arXiv:2504.13237) | SVD importance-aware delta sparsification |
| **Delta-CoMe** (arXiv:2406.08903) | Mixed-precision delta quantization |
| **ZipNN** (arXiv:2411.05239) | FP exponent skewness for neural compression |
| **Δ-Patching** (arXiv:2303.14772) | Lightweight weight patches |
| **Model Breadcrumbs** (arXiv:2312.06795) | Sparse weight differences for multi-task |

### Existing Tools

| Tool | Capability | Limitation |
|------|------------|------------|
| **TinyMLDelta** | TFLite Micro patches | Limited compression |
| **HDiffPatch** | Fast binary diff, 1KB RAM | Not tensor-aware |
| **ZipNN** | Neural compression | Not a diff tool |
| **bsdiff** | Standard binary diff | Slow, memory-hungry |

## Differences from Mithril

| Aspect | Mithril | Mallorn |
|--------|---------|----------|
| **Target user** | ML training engineers | Embedded/edge engineers |
| **Model size** | 100GB+ checkpoints | 200KB - 2GB models |
| **Formats** | safetensors, PyTorch | TFLite, GGUF, ONNX |
| **Update type** | Training checkpoints | Deployed model OTA |
| **Storage** | S3/GCS cloud | Flash, A/B slots |
| **Bandwidth** | Datacenter | LoRa, NB-IoT, cellular |
| **Constraints** | Compression ratio | Patch size, 1KB RAM |
| **Primary operation** | Compress/decompress | Diff/patch |

**Future integration:** Both projects may share compression primitives (ZipNN-style exponent coding) via a common `neural-compression` crate.

## Related Projects

- **Mithril** — Training checkpoint compression (same author)
- **HDiffPatch** — Generic binary diff (integration candidate)
- **ZipNN** — Neural compression (technique to adopt)
- **TinyMLDelta** — TFLite Micro patches (prior art)
- **Silicon Labs MLTK** — Model versioning metadata
