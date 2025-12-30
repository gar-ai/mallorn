# Mallorn Scope

What we're building, what we're deferring, and what we're never building.

## MVP (v0.1) — MUST HAVE

Core functionality to prove the concept works.

| Feature | Rationale | Hardware Req |
|---------|-----------|--------------|
| TFLite parser | Most common edge format | CPU only |
| GGUF parser | llama.cpp ecosystem | CPU only |
| Tensor-aware diff | Core value prop | CPU, <8GB RAM |
| Basic compression (zstd) | Baseline | CPU only |
| Patch application | Complete the loop | CPU only |
| SHA256 verification | Security baseline | CPU only |
| CLI (diff, patch, verify) | Developer UX | N/A |
| Invariant tests | Correctness guarantee | N/A |

**MVP Definition of Done:**
- [ ] `mallorn diff v1.tflite v2.tflite -o patch.tflp` works
- [ ] `mallorn patch v1.tflite patch.tflp -o v2.tflite` produces identical output
- [ ] Patch size < 10% of model size for minor weight updates
- [ ] All invariant tests pass
- [ ] Works on M3 Pro and 4080S dev machines

## v0.2 — Neural Compression

Improved compression using neural network structure awareness.

| Feature | Why Defer | Dependency |
|---------|-----------|------------|
| ZipNN-style exponent grouping | Research validation needed | v0.1 baseline |
| Byte-plane separation | Research validation needed | v0.1 baseline |
| Improved benchmarks | Need baseline first | v0.1 |
| Hypothesis tests | Need baseline to compare | v0.1 |

**v0.2 Definition of Done:**
- [ ] 20%+ improvement over v0.1 zstd-only compression
- [ ] Decompression speed >50 MB/s
- [ ] Hypothesis tests pass

## v0.3 — Embedded Support

Minimal C library for microcontrollers.

| Feature | Why Defer | Dependency |
|---------|-----------|------------|
| mallorn-lite C library | Core Rust must stabilize | v0.2 |
| Streaming patcher (1KB RAM) | Complex implementation | v0.2 |
| ESP32 example | Needs mallorn-lite | mallorn-lite |
| STM32 example | Needs mallorn-lite | mallorn-lite |
| A/B slot management | Needs mallorn-lite | mallorn-lite |

**v0.3 Definition of Done:**
- [ ] `mallorn_step()` loop works with 1KB buffer
- [ ] ESP32 example compiles and runs
- [ ] STM32 example compiles and runs
- [ ] Atomic update with rollback works

## v0.4 — Python & Polish

Developer experience improvements.

| Feature | Why Defer | Dependency |
|---------|-----------|------------|
| Python bindings (PyO3) | Core must stabilize | v0.3 |
| `pip install mallorn` | Needs Python bindings | Python bindings |
| ONNX support | Lower priority format | v0.2 |
| Documentation site | Need stable API | v0.3 |

**v0.4 Definition of Done:**
- [ ] `pip install mallorn` works on Linux/macOS/Windows
- [ ] Python API matches Rust API
- [ ] ONNX diff/patch works
- [ ] Docs published

## v1.0 — Production Ready

| Feature | Why Defer | Dependency |
|---------|-----------|------------|
| Cryptographic signatures | Security hardening | v0.4 |
| Downgrade protection | Security hardening | v0.4 |
| nRF52 example | Smallest MCU target | v0.3 |
| Performance optimization | Correctness first | v0.4 |
| Fuzzing | Security hardening | v0.4 |

---

## OUT OF SCOPE — NEVER

Features we will NOT build.

| Feature | Why Never |
|---------|-----------|
| **Model training** | Wrong tool (use PyTorch/JAX) |
| **Model serving** | Wrong tool (use vLLM/Triton) |
| **Full OTA protocol** | Too broad (use hawkBit/Mender) |
| **Cloud storage** | Wrong tool (use S3/GCS directly) |
| **Model conversion** | Wrong tool (use ONNX/TFLite converter) |
| **Multi-device coordination** | Out of scope (single device focus) |
| **GUI** | CLI + library is sufficient |
| **PyTorch/safetensors** | Use Mithril for training checkpoints |
| **Quantization** | Wrong tool (use llama.cpp/ONNX) |
| **Model optimization** | Wrong tool (use TensorRT/OpenVINO) |

---

## Scope Boundaries

### What Mallorn IS

```
┌─────────────────────────────────────────────────────────┐
│                      MALLORN                             │
│                                                          │
│  "Delta updates for deployed edge models"                │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Input: Model v1 + Model v2                      │    │
│  │ Output: Minimal patch file                       │    │
│  │ Apply: Model v1 + Patch → Model v2              │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  Formats: TFLite, GGUF, ONNX                            │
│  Targets: Edge devices with limited bandwidth           │
│  Constraint: 1KB RAM minimum for patch application      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### What Mallorn is NOT

```
┌─────────────────────────────────────────────────────────┐
│                    NOT MALLORN                           │
│                                                          │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Model Training  │  │ Model Serving   │              │
│  │ (PyTorch, JAX)  │  │ (vLLM, Triton)  │              │
│  └─────────────────┘  └─────────────────┘              │
│                                                          │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ OTA Protocol    │  │ Cloud Storage   │              │
│  │ (hawkBit)       │  │ (S3, GCS)       │              │
│  └─────────────────┘  └─────────────────┘              │
│                                                          │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Training Ckpts  │  │ Quantization    │              │
│  │ (Mithril)       │  │ (llama.cpp)     │              │
│  └─────────────────┘  └─────────────────┘              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Claude Code Feasibility Check

Before building each feature, verify:

- [ ] **No feature requires >2 weeks** of implementation
- [ ] **Reference implementations exist** to study
- [ ] **Testable on consumer hardware** (M3 Pro 18GB, 4080S 16GB)
- [ ] **Applying known techniques** (not novel research)
- [ ] **Clear success criteria** (measurable)

### MVP Feasibility Assessment

| Feature | Complexity | Reference Exists | HW Fits |
|---------|------------|------------------|---------|
| TFLite parser | Medium | ✓ flatbuffers | ✓ |
| GGUF parser | Medium | ✓ llama.cpp | ✓ |
| Tensor-aware diff | Medium | ✓ bsdiff | ✓ |
| Zstd compression | Low | ✓ zstd crate | ✓ |
| Patch application | Medium | ✓ bsdiff | ✓ |
| SHA256 | Low | ✓ sha2 crate | ✓ |
| CLI | Low | ✓ clap | ✓ |

**Assessment: MVP is feasible for Claude Code implementation.**

---

## Timeline

```
Week 1-2:   mallorn-core + TFLite parser
Week 3-4:   GGUF parser + tensor-aware diff
Week 5-6:   Patch format + CLI (MVP complete)
Week 7-8:   Neural compression (v0.2)
Week 9-12:  mallorn-lite C library (v0.3)
Week 13+:   Python bindings, ONNX (v0.4)
```

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2024-12 | TFLite before ONNX | More common in edge deployments |
| 2024-12 | GGUF in MVP | llama.cpp ecosystem growing fast |
| 2024-12 | No PyTorch support | Use Mithril for training checkpoints |
| 2024-12 | 1KB RAM target | Covers ESP32/STM32/nRF52 |
| 2024-12 | Rust + C (no C++) | Embedded compatibility |
