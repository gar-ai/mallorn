/**
 * Mallorn nRF52 Example
 *
 * Demonstrates delta model updates on nRF52 using mallorn-lite.
 * Uses internal flash with DFU-style dual-bank layout.
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "nrf.h"
#include "nrf_delay.h"
#include "nrf_gpio.h"
#include "nrf_log.h"
#include "nrf_log_ctrl.h"
#include "nrf_log_default_backends.h"
#include "nrf_fstorage.h"
#include "nrf_fstorage_nvmc.h"
#include "app_timer.h"
#include "nrf_pwr_mgmt.h"

#include "mallorn.h"

/* Flash layout for nRF52840 (1MB flash) */
#define MODEL_BANK_A_ADDR   0x40000   /* 256KB offset */
#define MODEL_BANK_B_ADDR   0x80000   /* 512KB offset */
#define MODEL_BANK_SIZE     0x40000   /* 256KB per bank */

/* Mallorn working buffer */
static uint8_t mallorn_buffer[1024];
static mallorn_patcher_t patcher;

/* Flash contexts */
typedef struct {
    uint32_t base;
    uint32_t offset;
    uint32_t size;
} flash_ctx_t;

static flash_ctx_t src_ctx, patch_ctx, out_ctx;

/* Active bank (stored in UICR) */
static uint8_t active_bank = 0;

/* fstorage instance */
NRF_FSTORAGE_DEF(nrf_fstorage_t fstorage) = {
    .evt_handler = NULL,
    .start_addr = MODEL_BANK_A_ADDR,
    .end_addr = MODEL_BANK_B_ADDR + MODEL_BANK_SIZE,
};

/**
 * Flash read callback
 */
static size_t flash_read_cb(uint8_t *ctx, uint8_t *buf, size_t max_len) {
    flash_ctx_t *fc = (flash_ctx_t *)ctx;

    size_t remaining = fc->size - fc->offset;
    size_t to_read = (remaining < max_len) ? remaining : max_len;

    if (to_read == 0) return 0;

    memcpy(buf, (void *)(fc->base + fc->offset), to_read);
    fc->offset += to_read;

    return to_read;
}

/**
 * Flash write callback
 */
static size_t flash_write_cb(uint8_t *ctx, const uint8_t *buf, size_t len) {
    flash_ctx_t *fc = (flash_ctx_t *)ctx;

    ret_code_t rc = nrf_fstorage_write(
        &fstorage,
        fc->base + fc->offset,
        buf,
        len,
        NULL
    );

    if (rc != NRF_SUCCESS) {
        NRF_LOG_ERROR("Flash write failed: %d", rc);
        return 0;
    }

    /* Wait for write to complete */
    while (nrf_fstorage_is_busy(&fstorage)) {
        nrf_pwr_mgmt_run();
    }

    fc->offset += len;
    return len;
}

/**
 * Erase flash bank
 */
static ret_code_t erase_bank(uint32_t addr) {
    ret_code_t rc = nrf_fstorage_erase(&fstorage, addr, MODEL_BANK_SIZE / 4096, NULL);
    if (rc != NRF_SUCCESS) return rc;

    while (nrf_fstorage_is_busy(&fstorage)) {
        nrf_pwr_mgmt_run();
    }
    return NRF_SUCCESS;
}

/**
 * Apply delta patch
 */
ret_code_t mallorn_apply_patch(
    const uint8_t *patch_data,
    uint32_t patch_size,
    const uint8_t *expected_hash
) {
    uint32_t src_addr = active_bank == 0 ? MODEL_BANK_A_ADDR : MODEL_BANK_B_ADDR;
    uint32_t dst_addr = active_bank == 0 ? MODEL_BANK_B_ADDR : MODEL_BANK_A_ADDR;

    NRF_LOG_INFO("Erasing target bank...");
    ret_code_t rc = erase_bank(dst_addr);
    if (rc != NRF_SUCCESS) return rc;

    /* Initialize mallorn */
    if (mallorn_init(&patcher, mallorn_buffer, sizeof(mallorn_buffer)) != OK) {
        return NRF_ERROR_INTERNAL;
    }

    /* Set up contexts */
    src_ctx = (flash_ctx_t){ .base = src_addr, .offset = 0, .size = MODEL_BANK_SIZE };
    patch_ctx = (flash_ctx_t){ .base = (uint32_t)patch_data, .offset = 0, .size = patch_size };
    out_ctx = (flash_ctx_t){ .base = dst_addr, .offset = 0, .size = MODEL_BANK_SIZE };

    mallorn_set_source(&patcher, flash_read_cb, (uint8_t *)&src_ctx);
    mallorn_set_patch(&patcher, flash_read_cb, (uint8_t *)&patch_ctx);
    mallorn_set_output(&patcher, flash_write_cb, (uint8_t *)&out_ctx);

    NRF_LOG_INFO("Applying patch...");
    enum mallorn_result_t result;
    uint32_t steps = 0;

    while ((result = mallorn_step(&patcher)) == CONTINUE) {
        steps++;
        if (steps % 100 == 0) {
            NRF_LOG_FLUSH();
        }
    }

    if (result != OK) {
        NRF_LOG_ERROR("Patch failed: %d", result);
        return NRF_ERROR_INTERNAL;
    }

    /* Verify */
    if (mallorn_verify(&patcher, expected_hash) != OK) {
        NRF_LOG_ERROR("Hash verification failed!");
        return NRF_ERROR_INVALID_DATA;
    }

    /* Switch bank */
    active_bank = 1 - active_bank;
    NRF_LOG_INFO("Success! Active bank: %c", 'A' + active_bank);

    return NRF_SUCCESS;
}

/**
 * Get active model address
 */
uint32_t mallorn_get_model_addr(void) {
    return active_bank == 0 ? MODEL_BANK_A_ADDR : MODEL_BANK_B_ADDR;
}

/**
 * Initialize
 */
static void mallorn_subsystem_init(void) {
    ret_code_t rc = nrf_fstorage_init(&fstorage, &nrf_fstorage_nvmc, NULL);
    APP_ERROR_CHECK(rc);

    /* Read active bank from retained register or flash */
    active_bank = 0;  /* Default to bank A */

    NRF_LOG_INFO("Mallorn initialized, active bank: %c", 'A' + active_bank);
}

int main(void) {
    ret_code_t rc;

    /* Initialize logging */
    rc = NRF_LOG_INIT(NULL);
    APP_ERROR_CHECK(rc);
    NRF_LOG_DEFAULT_BACKENDS_INIT();

    /* Initialize timer and power management */
    rc = app_timer_init();
    APP_ERROR_CHECK(rc);

    rc = nrf_pwr_mgmt_init();
    APP_ERROR_CHECK(rc);

    NRF_LOG_INFO("Mallorn nRF52 Example");
    NRF_LOG_INFO("Buffer: %d bytes (min: %zu)", sizeof(mallorn_buffer), mallorn_min_buffer_size());

    /* Initialize mallorn */
    mallorn_subsystem_init();

    /* Main loop */
    while (true) {
        NRF_LOG_FLUSH();
        nrf_pwr_mgmt_run();
    }
}
