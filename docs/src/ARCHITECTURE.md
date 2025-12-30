# Mallorn Architecture

Edge model delta updates for resource-constrained devices.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MALLORN ARCHITECTURE                         │
│                    "Grow and Spread Efficiently"                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   SERVER SIDE (Patch Generation)                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                                                              │  │
│   │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │  │
│   │   │ Model   │───▶│ Format  │───▶│ Tensor  │───▶│ Neural  │  │  │
│   │   │ Loader  │    │ Parser  │    │ Differ  │    │ Compress│  │  │
│   │   └─────────┘    └─────────┘    └─────────┘    └─────────┘  │  │
│   │        │              │              │              │        │  │
│   │        ▼              ▼              ▼              ▼        │  │
│   │   ┌─────────────────────────────────────────────────────┐   │  │
│   │   │              Patch File (.tflp, .ggup)              │   │  │
│   │   └─────────────────────────────────────────────────────┘   │  │
│   │                                                              │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                         OTA Transfer                                 │
│                      (LoRa/NB-IoT/Cell)                             │
│                              │                                       │
│   DEVICE SIDE (Patch Application)                                    │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                                                              │  │
│   │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │  │
│   │   │ Patch   │───▶│ Stream  │───▶│ Verify  │───▶│  A/B    │  │  │
│   │   │ Reader  │    │ Apply   │    │ Hash    │    │  Swap   │  │  │
│   │   └─────────┘    └─────────┘    └─────────┘    └─────────┘  │  │
│   │        │                              │                      │  │
│   │        ▼                              ▼                      │  │
│   │   ┌──────────┐                  ┌──────────┐                │  │
│   │   │ 1KB RAM  │                  │ Rollback │                │  │
│   │   │ Buffer   │                  │ on Fail  │                │  │
│   │   └──────────┘                  └──────────┘                │  │
│   │                                                              │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
mallorn/
├── Cargo.toml                    # Workspace root
├── README.md
├── docs/
│   ├── ARCHITECTURE.md           # This file
│   ├── INTERFACES.md             # Trait definitions
│   ├── SCOPE.md                  # MVP/defer/never
│   ├── SPEC.md                   # Full specification
│   ├── TESTING.md                # Test strategy
│   ├── METRICS.md                # Success criteria
│   ├── METHODOLOGY.md            # Research TDD approach
│   └── RESEARCH.md               # Papers and references
│
├── crates/
│   ├── mallorn-core/             # Shared primitives
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── compression.rs    # ZipNN-style neural compression
│   │   │   ├── diff.rs           # Binary diffing primitives
│   │   │   ├── hash.rs           # Checksums, verification
│   │   │   └── types.rs          # Shared types
│   │   └── Cargo.toml
│   │
│   ├── mallorn-tflite/           # TFLite format support
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── parser.rs         # FlatBuffer parsing
│   │   │   ├── differ.rs         # Tensor-aware diffing
│   │   │   ├── patcher.rs        # Patch application
│   │   │   └── format.rs         # .tflp patch format
│   │   └── Cargo.toml
│   │
│   ├── mallorn-gguf/             # GGUF format support
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── parser.rs         # GGUF metadata/tensor parsing
│   │   │   ├── differ.rs         # Quantized tensor diffing
│   │   │   ├── patcher.rs        # Patch application
│   │   │   └── format.rs         # .ggup patch format
│   │   └── Cargo.toml
│   │
│   ├── mallorn-onnx/             # ONNX format (P1)
│   │   └── ...
│   │
│   ├── mallorn-cli/              # Command-line interface
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── diff.rs           # `mallorn diff` command
│   │   │   ├── patch.rs          # `mallorn patch` command
│   │   │   ├── verify.rs         # `mallorn verify` command
│   │   │   └── info.rs           # `mallorn info` command
│   │   └── Cargo.toml
│   │
│   └── mallorn-python/           # Python bindings (PyO3)
│       ├── src/
│       │   └── lib.rs
│       ├── Cargo.toml
│       └── pyproject.toml
│
├── embedded/
│   └── mallorn-lite/             # Minimal C library for MCUs
│       ├── mallorn_lite.h
│       ├── mallorn_lite.c
│       ├── CMakeLists.txt
│       └── examples/
│           ├── esp32/
│           ├── stm32/
│           └── nrf52/
│
├── tests/
│   ├── invariants/               # Must-pass correctness tests
│   ├── hypotheses/               # Performance expectation tests
│   └── fixtures/                 # Test models and patches
│
├── benches/
│   ├── diff_bench.rs
│   ├── patch_bench.rs
│   └── compression_bench.rs
│
└── examples/
    ├── basic_diff.rs
    ├── streaming_patch.rs
    └── ota_simulation.rs
```

## Core Components

### mallorn-core

Shared primitives used by all format-specific crates.

| Module | Responsibility |
|--------|----------------|
| `compression.rs` | ZipNN-style neural compression (exponent/mantissa separation) |
| `diff.rs` | Low-level binary diff primitives (bsdiff-like) |
| `hash.rs` | SHA256 checksums, verification, Merkle trees |
| `types.rs` | `PatchHeader`, `TensorDelta`, `CompressionMethod` |

### mallorn-tflite

TensorFlow Lite model support.

| Module | Responsibility |
|--------|----------------|
| `parser.rs` | Parse TFLite FlatBuffer format |
| `differ.rs` | Tensor-aware diffing (align on layer boundaries) |
| `patcher.rs` | Apply patches to TFLite models |
| `format.rs` | .tflp patch file format |

### mallorn-gguf

GGUF (llama.cpp) model support.

| Module | Responsibility |
|--------|----------------|
| `parser.rs` | Parse GGUF metadata and tensor layout |
| `differ.rs` | Handle quantized tensor deltas (Q4, Q8, etc.) |
| `patcher.rs` | Apply patches to GGUF models |
| `format.rs` | .ggup patch file format |

### mallorn-lite (C)

Minimal implementation for microcontrollers.

| Function | Responsibility |
|----------|----------------|
| `mallorn_init()` | Initialize patcher with buffer |
| `mallorn_step()` | Process one chunk (streaming) |
| `mallorn_verify()` | Verify final hash |
| `mallorn_abort()` | Cancel and rollback |

## Data Flow

### Patch Generation (Server)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Old Model  │     │   New Model  │     │   Format     │
│   (v1.tflite)│     │   (v2.tflite)│     │   Parser     │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Tensor-Aware  │
                    │    Differ     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    Neural     │
                    │  Compression  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Patch File   │
                    │   (.tflp)     │
                    └───────────────┘
```

### Patch Application (Device)

```
┌──────────────┐     ┌──────────────┐
│   Old Model  │     │ Patch File   │
│   (Flash A)  │     │  (OTA RX)    │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────────────────┼────────────────────┐
                            │                    │
                            ▼                    ▼
                    ┌───────────────┐    ┌───────────────┐
                    │   Streaming   │    │    1KB RAM    │
                    │    Patcher    │◀───│    Buffer     │
                    └───────┬───────┘    └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    Verify     │
                    │   Checksum    │
                    └───────┬───────┘
                            │
                     ┌──────┴──────┐
                     │             │
                     ▼             ▼
              ┌───────────┐ ┌───────────┐
              │  Success  │ │  Failure  │
              │  Swap A/B │ │  Rollback │
              └───────────┘ └───────────┘
```

## Dependencies

### Rust Crates

```toml
[workspace.dependencies]
# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Binary formats
flatbuffers = "24"           # TFLite parsing
byteorder = "1"              # Endian handling

# Compression
zstd = "0.13"                # Base compression
lz4_flex = "0.11"            # Fast compression option

# Hashing
sha2 = "0.10"                # SHA256 for verification
xxhash-rust = "0.8"          # Fast hashing for dedup

# CLI
clap = { version = "4", features = ["derive"] }
indicatif = "0.17"           # Progress bars

# Python bindings
pyo3 = { version = "0.21", features = ["extension-module"] }

# Testing
criterion = "0.5"            # Benchmarks
proptest = "1"               # Property testing
```

### C Dependencies (mallorn-lite)

```c
// No external dependencies - pure C99
// Optional: mbedtls for SHA256 (or bring your own)
```

## Hardware Targets

### Server (Patch Generation)
- Any modern system
- M3 Pro / 4080S development machines
- CI: GitHub Actions runners

### Device (Patch Application)

| Class | Example | RAM | Flash |
|-------|---------|-----|-------|
| High-end Edge | Jetson Nano | 4GB | 16GB+ |
| Mid-range MCU | ESP32-S3 | 512KB | 8MB |
| Low-end MCU | STM32L4 | 256KB | 1MB |
| Tiny MCU | nRF52832 | 64KB | 512KB |

**Minimum requirement:** 1KB free RAM for streaming patcher.

## Development Phases

### Phase 1: Core + TFLite (Week 1-4)
- [ ] mallorn-core primitives
- [ ] TFLite parser
- [ ] Tensor-aware differ
- [ ] Basic patcher
- [ ] CLI (diff, patch, verify)

### Phase 2: GGUF (Week 5-6)
- [ ] GGUF parser
- [ ] Quantized tensor handling
- [ ] .ggup format

### Phase 3: Neural Compression (Week 7-8)
- [ ] ZipNN-style exponent grouping
- [ ] Improved compression ratios
- [ ] Benchmarks vs baseline

### Phase 4: Embedded (Week 9-12)
- [ ] mallorn-lite C library
- [ ] ESP32 example
- [ ] STM32 example
- [ ] Streaming patcher (1KB RAM)

### Phase 5: Polish (Week 13+)
- [ ] ONNX support
- [ ] Python bindings
- [ ] Documentation
- [ ] Release

## Security Considerations

### Patch Integrity
- SHA256 hash of target model in patch header
- Streaming verification (hash computed incrementally)
- Atomic swap (A/B partitions)

### Rollback Protection
- Version numbers in patch metadata
- Prevent downgrade attacks
- Cryptographic signatures (optional, via mbedtls)

### Memory Safety
- Rust for server-side (memory safe)
- Careful C for embedded (bounded buffers, no malloc in hot path)
