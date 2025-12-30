# Mallorn

> *"Their bark was silver and smooth, and their boughs somewhat upswept after the manner of the beech; but they never grew save in the land of Lórien."*

**Edge model delta updates.** Reduce OTA bandwidth from full model transfers to minimal patches.

## Why Mallorn?

| Scenario | Full Update | Delta Update | Savings |
|----------|-------------|--------------|---------|
| 200KB TFLite over LoRa | 6+ hours | 15 minutes | 96% |
| 2GB GGUF over cellular | $2-5 data | $0.10-0.25 | 95% |
| 500MB ONNX to 10K devices | 5PB transfer | 250TB | 95% |

## Features

- **Tensor-aware diffing** — Exploits model structure, not just binary similarity
- **Neural-optimized compression** — ZipNN-style FP exponent coding
- **Streaming patch application** — Works with 1KB RAM (MCU-friendly)
- **A/B slot management** — Atomic updates with rollback
- **Multiple formats** — TFLite, GGUF, ONNX

## Quick Start

```bash
# Create patch
mallorn diff model_v1.tflite model_v2.tflite -o update.tflp

# Apply patch
mallorn patch model_v1.tflite update.tflp -o model_v2.tflite

# Python
import mallorn
mallorn.create_patch("v1.tflite", "v2.tflite", "update.tflp")
```

## Supported Formats

| Format | Status | Patch Extension |
|--------|--------|-----------------|
| TFLite | v1.0 | `.tflp` |
| GGUF | v1.0 | `.ggup` |
| ONNX | v1.0 | `.onxp` |
| mallorn-lite (C) | v1.0 | Streaming |
| Core ML | Future | TBD |

## Architecture

```
Cloud/Server                        Edge Device
┌──────────┐                       ┌──────────┐
│ Model v1 │                       │ Model v1 │
│  (2GB)   │                       │ (Flash)  │
└────┬─────┘                       └──────────┘
     │                                   ▲
     ▼                                   │
┌──────────┐    OTA (50MB)         ┌────┴─────┐
│ Model v2 │ ─────────────────────▶│  Patch   │
│  (2GB)   │                       │ Applied  │
└────┬─────┘                       └──────────┘
     │                                   │
     ▼                                   ▼
┌──────────┐                       ┌──────────┐
│ Mallorn  │                       │ Model v2 │
│  Diff    │                       │ (Flash)  │
└──────────┘                       └──────────┘
```

## Embedded Integration

```c
#include "mallorn_lite.h"

// 1KB working memory
uint8_t buffer[1024];
mallorn_patcher_t patcher;
mallorn_init(&patcher, buffer, sizeof(buffer));

// Stream-based patch application
while (mallorn_step(&patcher) == MALLORN_CONTINUE) {
    watchdog_feed();
}
```

## Relationship to Mithril

| | Mithril | Mallorn |
|-|---------|---------|
| **For** | Training engineers | Embedded engineers |
| **Models** | 100GB+ checkpoints | 200KB-2GB deployed |
| **Storage** | S3/GCS | Flash, A/B slots |
| **Operation** | Compress | Diff/Patch |

Both projects may share compression primitives in the future.

## Documentation

- [Quick Start](./quick-start.md) - Get started in 5 minutes
- [CLI Reference](./cli.md) - Command-line usage
- [Python API](./python.md) - Python bindings
- [Embedded Integration](./embedded.md) - ESP32, STM32, nRF52
- [Security & Signing](./security.md) - Ed25519 signatures
- [Specification](./SPEC.md) - Full technical specification

## License

MIT OR Apache-2.0
