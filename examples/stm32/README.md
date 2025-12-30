# Mallorn STM32 Example

Delta model updates on STM32 using mallorn-lite with external QSPI flash.

## Features

- A/B slot scheme in external QSPI flash
- Streaming patch application (1KB RAM)
- SHA256 verification
- Watchdog-safe operation
- Backup register slot tracking

## Tested Platforms

- STM32F4 Discovery with QSPI flash
- STM32H7 Nucleo with external flash
- STM32L4 with OctoSPI

## Prerequisites

1. STM32CubeIDE or STM32CubeMX
2. Rust with ARM target:
   ```bash
   rustup target add thumbv7em-none-eabihf
   ```

## Building mallorn-lite for STM32

```bash
# From mallorn root
cargo build -p mallorn-lite --release --target thumbv7em-none-eabihf

# Copy library
cp target/thumbv7em-none-eabihf/release/libmallorn_lite.a \
   examples/stm32/Mallorn/
```

## Project Setup

1. Create STM32CubeIDE project for your MCU
2. Copy `Core/Src/main.c` and `Core/Inc/mallorn_config.h`
3. Copy `Mallorn/` folder with library and header
4. Add to include paths: `Mallorn/`
5. Link against: `libmallorn_lite.a`

## Memory Layout (External QSPI Flash)

| Address | Size | Purpose |
|---------|------|---------|
| 0x90000000 | 4MB | Model Slot A |
| 0x90400000 | 4MB | Model Slot B |

## Memory Usage

- mallorn_patcher_t: 256 bytes
- Working buffer: 1024 bytes
- Total: ~1.3KB RAM

## API

```c
void Mallorn_Init(void);
uint32_t Mallorn_GetActiveModelAddr(void);
HAL_StatusTypeDef Mallorn_ApplyPatch(
    const uint8_t *patch_data,
    uint32_t patch_size,
    const uint8_t *expected_hash
);
```

## License

MIT OR Apache-2.0
