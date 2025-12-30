# Quick Start

## Installation

### Python

```bash
pip install mallorn
```

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
mallorn-core = "1.0"
mallorn-tflite = "1.0"  # For TFLite models
mallorn-gguf = "1.0"    # For GGUF models
mallorn-onnx = "1.0"    # For ONNX models
```

### CLI

```bash
cargo install mallorn-cli
```

## Creating a Patch

### Command Line

```bash
# Create a patch between two model versions
mallorn diff model_v1.tflite model_v2.tflite -o update.tflp

# Apply the patch to recreate the new model
mallorn patch model_v1.tflite update.tflp -o model_v2_restored.tflite

# Verify patch integrity
mallorn verify model_v1.tflite update.tflp
```

### Python

```python
import mallorn

# Create a patch
stats = mallorn.create_patch(
    "model_v1.tflite",
    "model_v2.tflite",
    "update.tflp",
    compression_level=3
)
print(f"Patch size: {stats.patch_size} bytes")
print(f"Compression ratio: {stats.compression_ratio:.1f}x")

# Apply patch
verification = mallorn.apply_patch(
    "model_v1.tflite",
    "update.tflp",
    "model_v2_restored.tflite"
)
print(f"Valid: {verification.is_valid()}")

# Get patch info
info = mallorn.patch_info("update.tflp")
print(f"Format: {info.format}")
print(f"Modified tensors: {info.tensors_modified}")
```

### Rust

```rust
use mallorn_tflite::{TFLiteDiffer, TFLitePatcher, serialize_patch, deserialize_patch};
use mallorn_core::DiffOptions;

// Create a differ with options
let options = DiffOptions::default();
let differ = TFLiteDiffer::with_options(options);

// Compute diff
let patch = differ.diff_from_bytes(&old_model, &new_model)?;

// Serialize to file
let patch_bytes = serialize_patch(&patch)?;
std::fs::write("update.tflp", &patch_bytes)?;

// Apply patch
let patcher = TFLitePatcher::new();
let (new_model, verification) = patcher.apply_and_verify(&old_model, &patch)?;
```

## Supported Formats

| Format | Extension | Patch Extension |
|--------|-----------|-----------------|
| TFLite | `.tflite` | `.tflp` |
| GGUF   | `.gguf`   | `.ggup` |
| ONNX   | `.onnx`   | `.onxp` |

## Compression Options

```python
import mallorn

# Standard compression (default)
mallorn.create_patch("v1.tflite", "v2.tflite", "patch.tflp", compression_level=3)

# Maximum compression
mallorn.create_patch("v1.tflite", "v2.tflite", "patch.tflp", compression_level=19)

# Neural-optimized compression (for fine-tuned models)
mallorn.create_patch("v1.tflite", "v2.tflite", "patch.tflp", neural=True)
```

## Embedded Integration

For microcontrollers with limited RAM, use the C library:

```c
#include "mallorn_lite.h"

uint8_t buffer[1024];
mallorn_patcher_t patcher;

mallorn_init(&patcher, buffer, sizeof(buffer));

while (mallorn_step(&patcher) == MALLORN_CONTINUE) {
    watchdog_feed();
}

if (mallorn_verify(&patcher) == MALLORN_OK) {
    // Patch applied successfully
}
```

See the [ESP32](https://github.com/gabedavis/mallorn/tree/main/examples/esp32) and [STM32](https://github.com/gabedavis/mallorn/tree/main/examples/stm32) examples for complete implementations.
