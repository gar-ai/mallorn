/**
 * Mallorn STM32 Example
 *
 * Demonstrates delta model updates on STM32 using mallorn-lite.
 * Tested on STM32F4/STM32H7 with external QSPI flash.
 */

#include "main.h"
#include "mallorn.h"
#include <string.h>

/* Private variables */
static uint8_t mallorn_buffer[1024];
static mallorn_patcher_t patcher;

/* Flash contexts */
typedef struct {
    uint32_t base_addr;
    uint32_t offset;
    uint32_t size;
} flash_ctx_t;

static flash_ctx_t source_ctx;
static flash_ctx_t patch_ctx;
static flash_ctx_t output_ctx;

/* Model slot addresses in external QSPI flash */
#define MODEL_SLOT_A_ADDR  0x90000000
#define MODEL_SLOT_B_ADDR  0x90400000
#define MODEL_SLOT_SIZE    0x400000  /* 4MB per slot */

static uint8_t active_slot = 0;

/**
 * QSPI flash read callback
 */
static size_t qspi_read_cb(uint8_t *ctx, uint8_t *buf, size_t max_len) {
    flash_ctx_t *fc = (flash_ctx_t *)ctx;

    size_t remaining = fc->size - fc->offset;
    size_t to_read = (remaining < max_len) ? remaining : max_len;

    if (to_read == 0) return 0;

    /* Memory-mapped QSPI read */
    memcpy(buf, (void *)(fc->base_addr + fc->offset), to_read);
    fc->offset += to_read;

    return to_read;
}

/**
 * QSPI flash write callback
 */
static size_t qspi_write_cb(uint8_t *ctx, const uint8_t *buf, size_t len) {
    flash_ctx_t *fc = (flash_ctx_t *)ctx;

    /* Exit memory-mapped mode, write, re-enter */
    /* Implementation depends on your QSPI driver */
    if (BSP_QSPI_Write((uint8_t *)buf, fc->base_addr + fc->offset - 0x90000000, len) != QSPI_OK) {
        return 0;
    }

    fc->offset += len;
    return len;
}

/**
 * Apply delta patch to model
 */
HAL_StatusTypeDef Mallorn_ApplyPatch(
    const uint8_t *patch_data,
    uint32_t patch_size,
    const uint8_t *expected_hash
) {
    /* Determine source and target slots */
    uint32_t src_addr = active_slot == 0 ? MODEL_SLOT_A_ADDR : MODEL_SLOT_B_ADDR;
    uint32_t dst_addr = active_slot == 0 ? MODEL_SLOT_B_ADDR : MODEL_SLOT_A_ADDR;

    /* Erase target slot */
    if (BSP_QSPI_Erase_Block(dst_addr - 0x90000000) != QSPI_OK) {
        return HAL_ERROR;
    }

    /* Initialize mallorn patcher */
    if (mallorn_init(&patcher, mallorn_buffer, sizeof(mallorn_buffer)) != OK) {
        return HAL_ERROR;
    }

    /* Set up contexts */
    source_ctx.base_addr = src_addr;
    source_ctx.offset = 0;
    source_ctx.size = MODEL_SLOT_SIZE;

    patch_ctx.base_addr = (uint32_t)patch_data;
    patch_ctx.offset = 0;
    patch_ctx.size = patch_size;

    output_ctx.base_addr = dst_addr;
    output_ctx.offset = 0;
    output_ctx.size = MODEL_SLOT_SIZE;

    /* Configure callbacks */
    mallorn_set_source(&patcher, qspi_read_cb, (uint8_t *)&source_ctx);
    mallorn_set_patch(&patcher, qspi_read_cb, (uint8_t *)&patch_ctx);
    mallorn_set_output(&patcher, qspi_write_cb, (uint8_t *)&output_ctx);

    /* Apply patch */
    enum mallorn_result_t result;
    while ((result = mallorn_step(&patcher)) == CONTINUE) {
        HAL_IWDG_Refresh(&hiwdg);  /* Feed watchdog */
    }

    if (result != OK) {
        return HAL_ERROR;
    }

    /* Verify hash */
    if (mallorn_verify(&patcher, expected_hash) != OK) {
        return HAL_ERROR;
    }

    /* Switch active slot */
    active_slot = 1 - active_slot;

    /* Store in backup registers or flash */
    HAL_PWR_EnableBkUpAccess();
    HAL_RTCEx_BKUPWrite(&hrtc, RTC_BKP_DR0, active_slot);

    return HAL_OK;
}

/**
 * Get active model base address
 */
uint32_t Mallorn_GetActiveModelAddr(void) {
    return active_slot == 0 ? MODEL_SLOT_A_ADDR : MODEL_SLOT_B_ADDR;
}

/**
 * Initialize mallorn subsystem
 */
void Mallorn_Init(void) {
    /* Read active slot from backup register */
    HAL_PWR_EnableBkUpAccess();
    active_slot = HAL_RTCEx_BKUPRead(&hrtc, RTC_BKP_DR0) & 0x01;
}

int main(void) {
    HAL_Init();
    SystemClock_Config();

    /* Initialize peripherals */
    MX_GPIO_Init();
    MX_QUADSPI_Init();
    MX_USART2_UART_Init();
    MX_IWDG_Init();
    MX_RTC_Init();

    /* Initialize mallorn */
    Mallorn_Init();

    printf("Mallorn STM32 Example\r\n");
    printf("Buffer size: %d bytes\r\n", sizeof(mallorn_buffer));
    printf("Active model slot: %c\r\n", 'A' + active_slot);

    /* Main loop */
    while (1) {
        /* Your inference code here using Mallorn_GetActiveModelAddr() */
        HAL_Delay(1000);
    }
}
