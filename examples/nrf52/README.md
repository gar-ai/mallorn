# Mallorn nRF52 Example

Delta model updates on nRF52 using mallorn-lite with internal flash.

## Features

- Dual-bank layout in internal flash
- Streaming patch application (1KB RAM)
- SHA256 verification
- Power-management safe operation
- Minimal footprint for constrained devices

## Tested Platforms

- nRF52840 DK (1MB flash)
- nRF52832 DK (512KB flash)
- Adafruit Feather nRF52840

## Prerequisites

1. nRF5 SDK v17.1.0 or nRF Connect SDK
2. Rust with ARM target:
   ```bash
   rustup target add thumbv7em-none-eabihf
   ```

## Building mallorn-lite for nRF52

```bash
# From mallorn root
cargo build -p mallorn-lite --release --target thumbv7em-none-eabihf

# Copy library
cp target/thumbv7em-none-eabihf/release/libmallorn_lite.a \
   examples/nrf52/mallorn/
```

## Project Setup (nRF5 SDK)

1. Copy example to `nRF5_SDK/examples/mallorn/`
2. Add to Makefile:
   ```makefile
   INC_FOLDERS += mallorn
   LIB_FILES += mallorn/libmallorn_lite.a
   ```
3. Build: `make`

## Flash Layout (nRF52840)

| Address | Size | Purpose |
|---------|------|---------|
| 0x00000 | 256KB | Application |
| 0x40000 | 256KB | Model Bank A |
| 0x80000 | 256KB | Model Bank B |
| 0xC0000 | 256KB | Reserved |

## Memory Usage

- mallorn_patcher_t: 256 bytes
- Working buffer: 1024 bytes
- **Total: ~1.3KB RAM**

This is ideal for the nRF52's constrained RAM (256KB on nRF52840).

## API

```c
void mallorn_subsystem_init(void);
uint32_t mallorn_get_model_addr(void);
ret_code_t mallorn_apply_patch(
    const uint8_t *patch_data,
    uint32_t patch_size,
    const uint8_t *expected_hash
);
```

## BLE OTA Integration

For BLE-based patch delivery, integrate with:
- Nordic DFU (nrf_dfu)
- Custom GATT service for streaming patches

Example flow:
1. Receive patch chunks over BLE
2. Buffer in RAM or external flash
3. Call `mallorn_apply_patch()`
4. Verify and switch banks

## License

MIT OR Apache-2.0
