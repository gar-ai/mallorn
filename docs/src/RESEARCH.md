# Mallorn Research References

Papers, repos, and prior art for edge model delta updates.

## Core Papers

### Delta Compression for Neural Networks

| Paper | Key Insight | Applicability |
|-------|-------------|---------------|
| **ImPart (April 2025)** | SVD importance-aware sparsification. Top singular vectors get high precision, tail gets low precision. 2× compression vs uniform. | Direct application for delta encoding. |
| **Delta-CoMe (June 2024)** | Mixed-precision delta quantization by singular value magnitude. Weight importance varies. | Adaptive precision for patches. |
| **ZipNN (Nov 2024)** | FP weights have skewed exponent distributions. Separate exponent/mantissa coding. 33-50% lossless improvement, 80GB/s. | Neural-aware compression layer. |
| **Δ-Patching (2023)** | Efficient delta updates for edge deployment. Tensor-aware diffing beats binary. | Core algorithm validation. |
| **Model Breadcrumbs (2024)** | Sparse delta representation for federated updates. | Relevance to fine-tuning patches. |

### Binary Diffing

| Paper/Tool | Key Insight | Applicability |
|------------|-------------|---------------|
| **bsdiff** | Suffix sorting for binary diff. Industry standard baseline. | Baseline to beat. |
| **xdelta3** | VCDIFF format, streaming friendly. | Alternative baseline. |
| **HDiffPatch** | Minimal memory patching (1KB). HPatchLite for embedded. | Streaming patcher design. |
| **courgette (Chrome)** | Executable-aware diffing. Structure awareness matters. | Tensor-aware diffing concept. |

### Embedded OTA

| Paper/Tool | Key Insight | Applicability |
|------------|-------------|---------------|
| **TinyMLDelta** | Delta updates specifically for TinyML. A/B slot patterns. | Embedded strategy. |
| **MCUboot** | Secure bootloader with swap/move upgrade. | A/B slot management. |
| **hawkBit** | Enterprise OTA for IoT. Campaign management. | What NOT to build (too broad). |

## Key Techniques

### ZipNN Exponent Skewness

```
Neural network weights have NON-UNIFORM exponent distributions.

FP32 layout: [sign:1][exponent:8][mantissa:23]

Observation: Exponents cluster around small values.
- Most weights are small (exponent < 127)
- Mantissas are more random

Solution: Encode exponents separately with specialized codec.
- Exponents: High entropy reduction (33% smaller)
- Mantissas: Standard compression

Result: 33-50% better than generic zstd.
```

### ImPart Importance-Aware Delta

```
Not all weight changes matter equally.

Given: delta = new_weights - old_weights
Compute: U, S, V = SVD(delta)

Key insight: Top singular vectors capture important changes.
- Top 20%: 8-bit precision (captures 80% of info)
- Middle 30%: 4-bit precision
- Bottom 50%: 2-bit or skip

Result: 2× compression vs uniform precision.
```

### HDiffPatch Streaming

```
Patch application with minimal RAM.

Constraint: Only 1KB working buffer.

Approach:
1. Patch format designed for streaming
2. Operations specify exact offsets
3. Read old, read patch, write new in lockstep
4. Never need full model in memory

Result: Patch 100MB model with 1KB RAM.
```

### Tensor-Aware Diffing

```
Binary diff sees: [bytes...]
Tensor-aware sees: [tensor1][tensor2][tensor3]

Binary diff: Finds any matching bytes (slow, suboptimal)
Tensor-aware: Aligns on tensor boundaries (fast, optimal)

Why it matters:
- Tensors have semantic meaning
- Tensor headers change, data may not
- Same architecture = same tensor layout
- Delta per tensor = better compression

Result: 30%+ improvement over bsdiff.
```

## Format References

### TFLite

| Resource | Purpose |
|----------|---------|
| [TFLite FlatBuffer Schema](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs) | File format definition |
| [TFLite Model Analyzer](https://www.tensorflow.org/lite/guide/model_analyzer) | Understanding structure |
| [TFLite Interpreter](https://www.tensorflow.org/lite/guide/inference) | How models are loaded |

### GGUF

| Resource | Purpose |
|----------|---------|
| [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) | File format definition |
| [llama.cpp GGUF](https://github.com/ggerganov/llama.cpp) | Reference implementation |
| [Quantization Types](https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.h) | Q4, Q8, etc. formats |

### ONNX

| Resource | Purpose |
|----------|---------|
| [ONNX Spec](https://onnx.ai/onnx/repo-docs/IR.html) | File format definition |
| [ONNX Protobuf](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto) | Protobuf schema |

## Implementation References

### Rust Crates

| Crate | Purpose | Notes |
|-------|---------|-------|
| `flatbuffers` | TFLite parsing | Zero-copy access |
| `byteorder` | Endian handling | GGUF parsing |
| `zstd` | Compression | Baseline, well-optimized |
| `lz4_flex` | Fast compression | For speed-critical paths |
| `sha2` | Checksums | Verification |
| `bsdiff` | Binary diff | Baseline comparison |

### C Libraries

| Library | Purpose | Notes |
|---------|---------|-------|
| `miniz` | Tiny zlib | For embedded |
| `xxhash` | Fast hashing | For dedup |
| `mbedtls` | Crypto | Optional signatures |

## Related Projects

### Mithril — Training Checkpoint Compression

Separate project for **training infrastructure**, not deployment.

| Aspect | Mithril | Mallorn |
|--------|---------|---------|
| Target | Training engineers | Embedded engineers |
| Models | 100GB+ checkpoints | 200KB-2GB models |
| Formats | safetensors, PyTorch DCP | TFLite, GGUF, ONNX |
| Storage | S3/GCS | Flash, A/B slots |

**Shared:** Both use ZipNN-style compression primitives.

## Reading Order

### Getting Started (Week 1)
1. **ZipNN paper** — Core compression insight
2. **HDiffPatch README** — Streaming patcher design
3. **TFLite schema** — Format understanding

### Deep Dive (Week 2-3)
4. **ImPart paper** — Importance-aware compression
5. **Delta-CoMe paper** — Mixed precision deltas
6. **bsdiff source** — Baseline algorithm

### Optimization (Week 4+)
7. **Δ-Patching paper** — Tensor-aware techniques
8. **TinyMLDelta** — Embedded patterns
9. **MCUboot source** — A/B slot management

## Open Questions

### Research Questions
1. Can we beat ZipNN with tensor-structure awareness?
2. What's the optimal precision allocation for deltas?
3. How do quantized formats (Q4, Q8) affect diff ratios?

### Engineering Questions
1. Minimum viable streaming buffer size?
2. Flash wear leveling implications of frequent updates?
3. Atomic update guarantees on power loss?

## Arxiv Links

```
ZipNN: https://arxiv.org/abs/2411.05239
ImPart: https://arxiv.org/abs/2504.13237
Delta-CoMe: https://arxiv.org/abs/2406.08903
```

## Code Repositories

```
HDiffPatch: https://github.com/sisong/HDiffPatch
bsdiff: https://github.com/mendsley/bsdiff
llama.cpp: https://github.com/ggerganov/llama.cpp
TFLite: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite
MCUboot: https://github.com/mcu-tools/mcuboot
```
