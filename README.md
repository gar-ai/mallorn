# Mallorn

> *"Their bark was silver and smooth, and their boughs somewhat upswept after the manner of the beech; but they never grew save in the land of Lórien."*

**Edge model delta updates.** Reduce OTA bandwidth from full model transfers to minimal patches.

[![CI](https://github.com/user/mallorn/workflows/CI/badge.svg)](https://github.com/user/mallorn/actions)
[![Crates.io](https://img.shields.io/crates/v/mallorn-core.svg)](https://crates.io/crates/mallorn-core)
[![docs.rs](https://docs.rs/mallorn-core/badge.svg)](https://docs.rs/mallorn-core)
[![PyPI](https://img.shields.io/pypi/v/mallorn.svg)](https://pypi.org/project/mallorn/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Why Mallorn?

| Scenario | Full Update | Delta Update | Savings |
|----------|-------------|--------------|---------|
| 200KB TFLite over LoRa | 6+ hours | 15 minutes | 96% |
| 2GB GGUF over cellular | $2-5 data | $0.10-0.25 | 95% |
| 500MB ONNX to 10K devices | 5PB transfer | 250TB | 95% |

## Features

- **Tensor-aware diffing** — Exploits model structure, not just binary similarity
- **Neural-optimized compression** — ZipNN-style byte-plane separation
- **Streaming patch application** — Works with 1KB RAM (MCU-friendly)
- **Ed25519 signatures** — Secure OTA updates
- **Multiple formats** — TFLite, GGUF, ONNX, SafeTensors, OpenVINO, CoreML, TensorRT
- **Parallel compression** — Multi-threaded tensor processing (4-8x speedup)
- **Model fingerprinting** — Quick version detection (~10ms for any size)
- **Patch chains** — Incremental v1→v2→v3 updates with squash support
- **Dictionary compression** — Train custom dictionaries for 20-40% better ratios
- **GPU acceleration** — CUDA/Metal support via Candle for 10-50x neural compression speedup
- **WebAssembly** — Browser-compatible patching with `mallorn-wasm`
- **Async HTTP** — Non-blocking downloads with resume and concurrent multi-patch support

## Quick Start

### CLI

```bash
cargo install mallorn-cli

# Create patch (with parallel compression)
mallorn diff model_v1.tflite model_v2.tflite -o update.tflp --parallel

# Apply patch
mallorn patch model_v1.tflite update.tflp -o model_v2.tflite

# Quick model fingerprinting (~10ms)
mallorn fingerprint model.tflite

# Create patch chain (v1 → v2 → v3)
mallorn chain create v1.tflite v2.tflite v3.tflite -o updates.chain

# Apply chain to reach latest version
mallorn chain apply v1.tflite updates.chain -o latest.tflite

# Squash chain to single patch
mallorn chain squash updates.chain -o direct.tflp

# Download patch with resume support
mallorn download https://models.example.com/update.tflp -o update.tflp --resume
```

### Python

```bash
pip install mallorn
```

```python
import mallorn

# Create patch
stats = mallorn.create_patch("v1.tflite", "v2.tflite", "update.tflp")
print(f"Patch: {stats.patch_size} bytes ({stats.compression_ratio:.1f}x)")

# Apply patch
mallorn.apply_patch("v1.tflite", "update.tflp", "v2.tflite")
```

### Rust

```toml
[dependencies]
mallorn-core = "1.0"
mallorn-tflite = "1.0"
```

```rust
use mallorn_tflite::{TFLiteDiffer, TFLitePatcher};

let differ = TFLiteDiffer::new();
let patch = differ.diff_from_bytes(&old_model, &new_model)?;

let patcher = TFLitePatcher::new();
let new_model = patcher.apply(&old_model, &patch)?;
```

### Embedded (C)

```c
#include "mallorn.h"

uint8_t buffer[1024];
mallorn_patcher_t patcher;

mallorn_init(&patcher, buffer, sizeof(buffer));
mallorn_set_source(&patcher, flash_read, ctx);
mallorn_set_patch(&patcher, http_read, ctx);
mallorn_set_output(&patcher, flash_write, ctx);

while (mallorn_step(&patcher) == CONTINUE) {
    watchdog_reset();
}
```

## Supported Formats

| Format | Crate | Patch Extension |
|--------|-------|-----------------|
| TFLite | `mallorn-tflite` | `.tflp` |
| GGUF | `mallorn-gguf` | `.ggup` |
| ONNX | `mallorn-onnx` | `.onxp` |
| SafeTensors | `mallorn-safetensors` | `.sftp` |
| OpenVINO | `mallorn-openvino` | `.ovip` |
| CoreML | `mallorn-coreml` | `.cmlp` |
| TensorRT | `mallorn-tensorrt` | `.trtp` |

## Performance

Benchmarked on Apple M1:

| Operation | Speed |
|-----------|-------|
| LZ4 decompress | 22 GB/s |
| Zstd decompress | 3.3 GB/s |
| Neural decompress | 230 MB/s |
| Streaming RAM | 256B + 1KB buffer |

## Project Structure

```
mallorn/
├── crates/
│   ├── mallorn-core/       # Core types, compression, signatures
│   ├── mallorn-tflite/     # TensorFlow Lite support
│   ├── mallorn-gguf/       # GGUF (llama.cpp) support
│   ├── mallorn-onnx/       # ONNX support
│   ├── mallorn-safetensors/# SafeTensors support
│   ├── mallorn-openvino/   # OpenVINO IR support
│   ├── mallorn-coreml/     # CoreML support
│   ├── mallorn-tensorrt/   # TensorRT support
│   ├── mallorn-cli/        # Command-line tool
│   ├── mallorn-python/     # Python bindings (PyO3)
│   ├── mallorn-lite/       # C library for embedded
│   ├── mallorn-candle/     # GPU acceleration (CUDA/Metal)
│   └── mallorn-wasm/       # WebAssembly bindings
├── examples/
│   ├── esp32/              # ESP-IDF example
│   ├── stm32/              # STM32 HAL example
│   ├── nrf52/              # nRF5 SDK example
│   ├── python/             # Python notebook & FastAPI server
│   └── browser/            # WebAssembly demo
├── docs/                   # mdbook documentation
└── fuzz/                   # cargo-fuzz targets
```

## Documentation

- [Quick Start](docs/src/quick-start.md)
- [CLI Reference](docs/src/cli.md)
- [Python API](docs/src/python.md)
- [Embedded Integration](docs/src/embedded.md)
- [Security & Signing](docs/src/security.md)
- [Patch Chains](docs/src/chains.md)
- [Model Fingerprinting](docs/src/fingerprinting.md)
- [Downloads & Manifests](docs/src/downloads.md)
- [Advanced Topics](docs/src/advanced.md)
- [Specification](docs/src/SPEC.md)

Build the documentation locally:

```bash
cd docs && mdbook serve
```

## Building

```bash
# All crates
cargo build --release

# Run tests
cargo test --all

# Run benchmarks
cargo bench -p mallorn-core

# Build C library
cargo build -p mallorn-lite --release
# Output: target/release/libmallorn_lite.a

# Build Python wheel
cd crates/mallorn-python && maturin build --release

# Build WebAssembly
cd crates/mallorn-wasm && wasm-pack build --target web

# Enable async HTTP downloads
cargo build -p mallorn-core --features async
```

## License

MIT OR Apache-2.0
