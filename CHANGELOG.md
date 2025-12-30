# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-29

### Added

#### v1.5 Features
- **GPU Acceleration** (`mallorn-candle`)
  - CUDA and Metal support via Candle framework
  - 10-50x speedup for neural compression operations
  - Sparse tensor encoding with GPU acceleration
  - Block-aligned delta computation
- **WebAssembly** (`mallorn-wasm`)
  - Browser-compatible model patching
  - Works with ArrayBuffer/Uint8Array inputs
  - Zstd and LZ4 compression support
  - Model fingerprinting in the browser
- **Async HTTP** (`mallorn-core` with `async` feature)
  - Non-blocking downloads with tokio
  - Concurrent multi-patch downloads
  - Resume support for interrupted transfers
- **CI/CD Pipeline**
  - GitHub Actions for testing and linting
  - Nightly fuzz testing (16 targets)
  - Cross-platform builds (Linux, macOS, Windows)
- **Documentation**
  - Patch chains guide
  - Model fingerprinting reference
  - Downloads and manifests documentation
  - Advanced topics (quantized deltas, streaming)
- **Examples**
  - Python Jupyter notebook with visualizations
  - FastAPI server example
  - ESP32 fingerprint API example

#### v1.4 Features
- **CLI Tool** (`mallorn-cli`)
  - `diff` - Create patches between model versions
  - `patch` - Apply patches to models
  - `verify` - Verify patch integrity
  - `info` - Display patch metadata
  - `keygen` - Generate Ed25519 keypairs
  - `sign` - Sign patches for secure OTA
  - `fingerprint` - Quick model version detection
  - `download` - HTTP downloads with resume
  - `chain` - Patch chain management
- **HTTP Downloads**
  - Resume support via Range headers
  - Hash verification
  - Progress callbacks
- **Model Fingerprinting**
  - ~10ms for any file size (64KB header + 4KB tail sampling)
  - Compact string representation
  - Fingerprint database for version management
- **Streaming Patch Application**
  - Memory-efficient patching
  - Configurable chunk sizes
  - Works with 1KB RAM on MCUs

#### v1.0-1.3 Features
- **Model Format Support**
  - TFLite (`mallorn-tflite`)
  - GGUF (`mallorn-gguf`)
  - ONNX (`mallorn-onnx`)
  - SafeTensors (`mallorn-safetensors`)
  - OpenVINO (`mallorn-openvino`)
  - CoreML (`mallorn-coreml`)
  - TensorRT (`mallorn-tensorrt`)
- **Neural-Optimized Compression**
  - Byte-plane separation (ZipNN-style)
  - F16 weight optimization
  - Sparse tensor detection
- **Dictionary Compression**
  - Custom dictionary training
  - 20-40% better ratios for model families
- **Patch Chains**
  - Incremental v1 -> v2 -> v3 updates
  - Chain squashing to single patch
  - Update path computation
- **Ed25519 Signatures**
  - Secure OTA updates
  - Trusted key management
  - Downgrade protection with version metadata
- **Parallel Compression**
  - Multi-threaded tensor processing
  - 4-8x speedup on multi-core systems
  - Adaptive thread count
- **Python Bindings** (`mallorn-python`)
  - PyO3-based native module
  - Full API coverage
  - Type stubs for IDE support
- **C Library** (`mallorn-lite`)
  - Embedded-friendly API
  - Streaming callbacks
  - No dynamic allocation
- **Quantized Delta Support**
  - Q4_0, Q4_K, Q5_K, Q6_K block types
  - Scale/min separation for better compression
  - GGUF quantization-aware diffing

### Performance

Benchmarked on Apple M1:

| Operation | Throughput |
|-----------|------------|
| LZ4 compression | 20+ GB/s |
| LZ4 decompression | 25+ GB/s |
| Zstd L1 compression | 5-6 GB/s |
| Zstd decompression | 3-4 GB/s |
| Neural compression | 200-400 MB/s |
| Model fingerprinting | 80+ GB/s (100MB file) |
| XOR delta compute | 2+ GB/s |

### Crate Versions

| Crate | Version |
|-------|---------|
| mallorn-core | 0.1.0 |
| mallorn-tflite | 0.1.0 |
| mallorn-gguf | 0.1.0 |
| mallorn-onnx | 0.1.0 |
| mallorn-safetensors | 0.1.0 |
| mallorn-openvino | 0.1.0 |
| mallorn-coreml | 0.1.0 |
| mallorn-tensorrt | 0.1.0 |
| mallorn-cli | 0.1.0 |
| mallorn-python | 0.1.0 |
| mallorn-lite | 0.1.0 |
| mallorn-candle | 0.1.0 |
| mallorn-wasm | 0.1.0 |
