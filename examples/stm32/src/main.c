/**
 * Mallorn OTA Delta Update Example for STM32
 *
 * This example demonstrates using mallorn-lite to apply delta patches
 * on an STM32 microcontroller with minimal RAM usage (1KB buffer).
 *
 * Memory Layout (example for STM32F4 with 1MB flash):
 * - 0x08000000-0x0801FFFF: Bootloader (128KB)
 * - 0x08020000-0x0807FFFF: Model V1 (384KB)
 * - 0x08080000-0x080DFFFF: Model V2 (384KB)
 * - 0x080E0000-0x080FFFFF: Patch storage (128KB)
 */

#include <stdint.h>
#include <string.h>
#include "mallorn.h"
#include "flash_io.h"

// Flash addresses (adjust for your MCU)
#define FLASH_MODEL_V1_ADDR  0x08020000
#define FLASH_MODEL_V1_SIZE  (384 * 1024)
#define FLASH_MODEL_V2_ADDR  0x08080000
#define FLASH_MODEL_V2_SIZE  (384 * 1024)
#define FLASH_PATCH_ADDR     0x080E0000
#define FLASH_PATCH_SIZE     (128 * 1024)

// Context for flash I/O callbacks
typedef struct {
    uint32_t base_addr;
    uint32_t offset;
    uint32_t size;
} flash_io_ctx_t;

// Read callback - reads from flash
static size_t flash_read(uint8_t *ctx_ptr, uint8_t *buf, size_t max_len)
{
    flash_io_ctx_t *ctx = (flash_io_ctx_t *)ctx_ptr;

    size_t remaining = ctx->size - ctx->offset;
    size_t to_read = (max_len < remaining) ? max_len : remaining;

    if (to_read == 0) {
        return 0;
    }

    // Direct memory-mapped read
    const uint8_t *src = (const uint8_t *)(ctx->base_addr + ctx->offset);
    memcpy(buf, src, to_read);

    ctx->offset += to_read;
    return to_read;
}

// Write callback - writes to flash
static size_t flash_write(uint8_t *ctx_ptr, const uint8_t *buf, size_t len)
{
    flash_io_ctx_t *ctx = (flash_io_ctx_t *)ctx_ptr;

    uint32_t addr = ctx->base_addr + ctx->offset;

    // Flash write (platform-specific)
    if (flash_program(addr, buf, len) != 0) {
        return 0;
    }

    ctx->offset += len;
    return len;
}

// Watchdog reset (platform-specific)
extern void watchdog_reset(void);

// Apply delta patch
int apply_patch(const uint8_t expected_hash[32])
{
    // Working buffer (1KB minimum)
    static uint8_t buffer[1024];

    // I/O contexts
    flash_io_ctx_t source_ctx = {
        .base_addr = FLASH_MODEL_V1_ADDR,
        .offset = 0,
        .size = FLASH_MODEL_V1_SIZE
    };

    flash_io_ctx_t patch_ctx = {
        .base_addr = FLASH_PATCH_ADDR,
        .offset = 0,
        .size = FLASH_PATCH_SIZE
    };

    flash_io_ctx_t output_ctx = {
        .base_addr = FLASH_MODEL_V2_ADDR,
        .offset = 0,
        .size = FLASH_MODEL_V2_SIZE
    };

    // Erase output region before patching
    if (flash_erase(FLASH_MODEL_V2_ADDR, FLASH_MODEL_V2_SIZE) != 0) {
        return -1;
    }

    // Initialize patcher
    mallorn_patcher_t patcher;
    enum mallorn_result_t result = mallorn_init(&patcher, buffer, sizeof(buffer));
    if (result != OK) {
        return -2;
    }

    // Configure I/O
    mallorn_set_source(&patcher, flash_read, (uint8_t *)&source_ctx);
    mallorn_set_patch(&patcher, flash_read, (uint8_t *)&patch_ctx);
    mallorn_set_output(&patcher, flash_write, (uint8_t *)&output_ctx);

    // Process patch step by step
    uint32_t step_count = 0;
    while ((result = mallorn_step(&patcher)) == CONTINUE) {
        step_count++;

        // Reset watchdog periodically
        if (step_count % 50 == 0) {
            watchdog_reset();
        }
    }

    if (result != OK) {
        mallorn_abort(&patcher);
        return -3;
    }

    // Verify hash
    result = mallorn_verify(&patcher, expected_hash);
    if (result != OK) {
        return -4;
    }

    return 0;  // Success
}

// Update boot configuration to use new model
int swap_to_new_model(void)
{
    // Platform-specific: update boot flag in backup registers,
    // option bytes, or dedicated flash sector
    return 0;
}

// Rollback to previous model
int rollback_to_old_model(void)
{
    // Platform-specific: restore boot flag
    return 0;
}

int main(void)
{
    // Platform initialization
    // SystemInit();
    // HAL_Init();

    // Example: Apply pre-downloaded patch
    // In production:
    // 1. Receive patch via UART/USB/network
    // 2. Store to patch flash region
    // 3. Apply patch
    // 4. Verify and swap

    uint8_t expected_hash[32] = {0};  // Would come from update manifest

    int result = apply_patch(expected_hash);
    if (result == 0) {
        swap_to_new_model();
        // Reset to boot with new model
        // NVIC_SystemReset();
    } else {
        // Patch failed, stay on old model
        rollback_to_old_model();
    }

    while (1) {
        // Main loop
    }

    return 0;
}
