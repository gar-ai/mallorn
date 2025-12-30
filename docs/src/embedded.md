# Embedded Integration

Mallorn provides a lightweight C library (`mallorn-lite`) designed for microcontrollers with limited RAM. This guide covers integration with ESP32, STM32, and nRF52 platforms.

## Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 1.3 KB | 2 KB |
| Flash | 32 KB | 64 KB |
| Stack | 256 bytes | 512 bytes |

The patcher struct itself is only 256 bytes and can be stack-allocated.

## C API Overview

```c
#include "mallorn.h"

// Initialize patcher with working buffer
mallorn_patcher_t patcher;
uint8_t buffer[1024];
mallorn_init(&patcher, buffer, sizeof(buffer));

// Set I/O callbacks
mallorn_set_source(&patcher, read_source_fn, source_ctx);
mallorn_set_patch(&patcher, read_patch_fn, patch_ctx);
mallorn_set_output(&patcher, write_output_fn, output_ctx);

// Apply patch in steps (watchdog-safe)
while (mallorn_step(&patcher) == CONTINUE) {
    watchdog_reset();
}

// Verify result
if (mallorn_verify(&patcher, expected_hash) == OK) {
    // Success!
}
```

## Callback Functions

### Read Callback

```c
typedef size_t (*mallorn_read_fn)(uint8_t *ctx, uint8_t *buf, size_t max_len);
```

Return the number of bytes read, or 0 on EOF/error.

### Write Callback

```c
typedef size_t (*mallorn_write_fn)(uint8_t *ctx, const uint8_t *buf, size_t len);
```

Return the number of bytes written.

## A/B Slot Pattern

For atomic updates with rollback capability:

```
Flash Layout:
┌─────────────────┐
│ Bootloader      │
├─────────────────┤
│ Model Slot A    │  ← Active
├─────────────────┤
│ Model Slot B    │  ← Standby (patch target)
├─────────────────┤
│ NVS / Config    │  ← Stores active slot
└─────────────────┘
```

1. Patch is applied to the standby slot
2. On success, switch active slot pointer
3. On boot failure, revert to previous slot

## Platform Examples

### ESP32 (ESP-IDF)

```c
#include "esp_partition.h"
#include "mallorn.h"

// Flash read callback
static size_t flash_read(uint8_t *ctx, uint8_t *buf, size_t max_len) {
    flash_ctx_t *f = (flash_ctx_t *)ctx;
    size_t to_read = MIN(max_len, f->remaining);
    if (to_read == 0) return 0;

    esp_partition_read(f->partition, f->offset, buf, to_read);
    f->offset += to_read;
    f->remaining -= to_read;
    return to_read;
}

// Flash write callback
static size_t flash_write(uint8_t *ctx, const uint8_t *buf, size_t len) {
    flash_ctx_t *f = (flash_ctx_t *)ctx;
    esp_partition_write(f->partition, f->offset, buf, len);
    f->offset += len;
    return len;
}
```

See [examples/esp32/](https://github.com/gabedavis/mallorn/tree/main/examples/esp32) for the complete implementation with:
- A/B partition scheme
- HTTP patch download
- NVS slot persistence
- Watchdog safety

### STM32 (HAL)

```c
#include "stm32f4xx_hal.h"
#include "mallorn.h"

// QSPI flash read
static size_t qspi_read(uint8_t *ctx, uint8_t *buf, size_t max_len) {
    qspi_ctx_t *q = (qspi_ctx_t *)ctx;
    HAL_QSPI_Receive(&hqspi, buf, max_len, HAL_MAX_DELAY);
    return max_len;
}

// QSPI flash write
static size_t qspi_write(uint8_t *ctx, const uint8_t *buf, size_t len) {
    qspi_ctx_t *q = (qspi_ctx_t *)ctx;
    HAL_QSPI_Transmit(&hqspi, (uint8_t *)buf, len, HAL_MAX_DELAY);
    return len;
}
```

See [examples/stm32/](https://github.com/gabedavis/mallorn/tree/main/examples/stm32) for:
- STM32CubeIDE project structure
- External QSPI flash support
- Backup register slot tracking

### nRF52 (nRF5 SDK)

```c
#include "nrf_fstorage.h"
#include "mallorn.h"

// Internal flash read
static size_t nrf_read(uint8_t *ctx, uint8_t *buf, size_t max_len) {
    nrf_ctx_t *n = (nrf_ctx_t *)ctx;
    memcpy(buf, (void *)n->address, max_len);
    n->address += max_len;
    return max_len;
}

// Internal flash write (via fstorage)
static size_t nrf_write(uint8_t *ctx, const uint8_t *buf, size_t len) {
    nrf_ctx_t *n = (nrf_ctx_t *)ctx;
    nrf_fstorage_write(&fstorage, n->address, buf, len, NULL);
    n->address += len;
    return len;
}
```

See [examples/nrf52/](https://github.com/gabedavis/mallorn/tree/main/examples/nrf52) for:
- Dual-bank flash layout
- BLE OTA integration hints
- Power-safe writes

## Building mallorn-lite

### As a Static Library

```bash
cargo build -p mallorn-lite --release

# Output:
# target/release/libmallorn_lite.a
# crates/mallorn-lite/include/mallorn.h
```

### Cross-Compilation

```bash
# For ARM Cortex-M
rustup target add thumbv7em-none-eabihf
cargo build -p mallorn-lite --release --target thumbv7em-none-eabihf

# For ESP32 (via esp-idf)
# Use the ESP-IDF component integration instead
```

## Error Handling

```c
enum mallorn_result_t {
    OK = 0,                      // Success
    CONTINUE = 1,                // Call step() again
    ERROR_INVALID_PATCH = -1,    // Patch format error
    ERROR_HASH_MISMATCH = -2,    // Verification failed
    ERROR_BUFFER_TOO_SMALL = -3, // Need at least 1KB
    ERROR_IO = -4,               // Read/write failed
    ERROR_ABORTED = -5,          // Aborted by user
};
```

Always check return values and implement proper error recovery.

## Memory Layout

```
mallorn_patcher_t (256 bytes, stack-allocatable):
┌──────────────────────────────┐
│ State machine (8 bytes)      │
│ I/O callbacks (24 bytes)     │
│ Hash state (104 bytes)       │
│ Decompressor state (80 bytes)│
│ Buffer pointers (40 bytes)   │
└──────────────────────────────┘

Working buffer (1KB minimum):
┌──────────────────────────────┐
│ Input staging (512 bytes)    │
│ Output staging (512 bytes)   │
└──────────────────────────────┘
```

## Best Practices

1. **Always verify** - Call `mallorn_verify()` before switching slots
2. **Keep old slot** - Don't erase until new model is validated
3. **Pet the watchdog** - Call watchdog reset in the step loop
4. **Handle power loss** - Assume writes can be interrupted
5. **Test rollback** - Verify your A/B switch logic works
