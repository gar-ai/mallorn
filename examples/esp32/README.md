# Mallorn ESP32 Example

Delta model updates on ESP32 using mallorn-lite.

## Features

- A/B partition scheme for atomic updates
- Streaming patch application (1KB RAM)
- SHA256 verification
- HTTP patch download
- NVS-backed slot tracking

## Prerequisites

1. ESP-IDF v5.0+ installed
2. Rust with ESP32 target:
   ```bash
   rustup target add xtensa-esp32-espidf
   ```

## Building mallorn-lite for ESP32

```bash
# From mallorn root
cargo build -p mallorn-lite --release --target xtensa-esp32-espidf

# Copy library
cp target/xtensa-esp32-espidf/release/libmallorn_lite.a \
   examples/esp32/components/mallorn/lib/
```

## Building the Example

```bash
cd examples/esp32
idf.py set-target esp32
idf.py build
idf.py flash monitor
```

## Partition Layout

| Partition | Size | Purpose |
|-----------|------|---------|
| model_a   | 2MB  | Model slot A |
| model_b   | 2MB  | Model slot B |

## Usage

1. Flash initial model to `model_a` partition
2. Generate patch: `mallorn diff old.tflite new.tflite -o patch.tflp`
3. Host patch on HTTP server
4. Call `apply_model_patch(url, expected_hash)`

## Memory Usage

- mallorn_patcher_t: 256 bytes (stack)
- Working buffer: 1024 bytes (stack)
- Total: ~1.3KB RAM for patching

## License

MIT OR Apache-2.0
