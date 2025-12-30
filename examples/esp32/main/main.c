/**
 * Mallorn ESP32 Example
 *
 * Demonstrates delta model updates on ESP32 using mallorn-lite.
 * Uses A/B partition scheme for atomic updates with rollback.
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "nvs_flash.h"
#include "esp_http_client.h"
#include "esp_task_wdt.h"

#include "mallorn.h"
#include "model_update.h"

static const char *TAG = "mallorn";

#define MODEL_PARTITION_A "model_a"
#define MODEL_PARTITION_B "model_b"

static uint8_t mallorn_buffer[1024];
static int active_slot = 0;

// Flash read callback
static size_t flash_read_cb(uint8_t *ctx, uint8_t *buf, size_t max_len) {
    flash_reader_t *r = (flash_reader_t *)ctx;
    size_t remaining = r->size - r->offset;
    size_t to_read = (remaining < max_len) ? remaining : max_len;
    if (to_read == 0) return 0;
    
    if (esp_partition_read(r->partition, r->offset, buf, to_read) != ESP_OK) {
        return 0;
    }
    r->offset += to_read;
    return to_read;
}

// Flash write callback  
static size_t flash_write_cb(uint8_t *ctx, const uint8_t *buf, size_t len) {
    flash_writer_t *w = (flash_writer_t *)ctx;
    if (esp_partition_write(w->partition, w->offset, buf, len) != ESP_OK) {
        return 0;
    }
    w->offset += len;
    return len;
}

// HTTP read callback
static size_t http_read_cb(uint8_t *ctx, uint8_t *buf, size_t max_len) {
    http_reader_t *r = (http_reader_t *)ctx;
    int len = esp_http_client_read(r->client, (char *)buf, max_len);
    return (len < 0) ? 0 : (size_t)len;
}

esp_err_t apply_model_patch(const char *patch_url, const uint8_t *expected_hash) {
    const char *src_name = active_slot == 0 ? MODEL_PARTITION_A : MODEL_PARTITION_B;
    const char *dst_name = active_slot == 0 ? MODEL_PARTITION_B : MODEL_PARTITION_A;

    const esp_partition_t *src = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, src_name);
    const esp_partition_t *dst = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, dst_name);

    if (!src || !dst) {
        ESP_LOGE(TAG, "Partitions not found");
        return ESP_ERR_NOT_FOUND;
    }

    ESP_LOGI(TAG, "Erasing %s...", dst_name);
    ESP_ERROR_CHECK(esp_partition_erase_range(dst, 0, dst->size));

    esp_http_client_config_t cfg = { .url = patch_url, .timeout_ms = 30000 };
    esp_http_client_handle_t client = esp_http_client_init(&cfg);
    ESP_ERROR_CHECK(esp_http_client_open(client, 0));
    esp_http_client_fetch_headers(client);

    mallorn_patcher_t patcher;
    if (mallorn_init(&patcher, mallorn_buffer, sizeof(mallorn_buffer)) != OK) {
        esp_http_client_cleanup(client);
        return ESP_FAIL;
    }

    flash_reader_t src_ctx = { .partition = src, .offset = 0, .size = src->size };
    http_reader_t patch_ctx = { .client = client };
    flash_writer_t dst_ctx = { .partition = dst, .offset = 0 };

    mallorn_set_source(&patcher, flash_read_cb, (uint8_t *)&src_ctx);
    mallorn_set_patch(&patcher, http_read_cb, (uint8_t *)&patch_ctx);
    mallorn_set_output(&patcher, flash_write_cb, (uint8_t *)&dst_ctx);

    ESP_LOGI(TAG, "Applying patch...");
    enum mallorn_result_t result;
    uint32_t steps = 0;

    while ((result = mallorn_step(&patcher)) == CONTINUE) {
        if (++steps % 100 == 0) esp_task_wdt_reset();
    }

    esp_http_client_cleanup(client);

    if (result != OK) {
        ESP_LOGE(TAG, "Patch failed: %d", result);
        return ESP_FAIL;
    }

    if (mallorn_verify(&patcher, expected_hash) != OK) {
        ESP_LOGE(TAG, "Hash mismatch!");
        return ESP_ERR_INVALID_CRC;
    }

    active_slot = 1 - active_slot;
    nvs_handle_t nvs;
    if (nvs_open("mallorn", NVS_READWRITE, &nvs) == ESP_OK) {
        nvs_set_i32(nvs, "slot", active_slot);
        nvs_commit(nvs);
        nvs_close(nvs);
    }

    ESP_LOGI(TAG, "Success! Active slot: %c", 'A' + active_slot);
    return ESP_OK;
}

// Get fingerprint of currently active model
esp_err_t get_model_fingerprint(mallorn_fingerprint_t *fp) {
    const char *name = active_slot == 0 ? MODEL_PARTITION_A : MODEL_PARTITION_B;
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, name);

    if (!part) {
        ESP_LOGE(TAG, "Partition not found");
        return ESP_ERR_NOT_FOUND;
    }

    flash_reader_t ctx = { .partition = part, .offset = 0, .size = part->size };

    if (mallorn_fingerprint(flash_read_cb, (uint8_t *)&ctx, part->size, fp) != OK) {
        ESP_LOGE(TAG, "Fingerprint failed");
        return ESP_FAIL;
    }

    return ESP_OK;
}

// Log fingerprint short ID as hex
void log_fingerprint(const mallorn_fingerprint_t *fp) {
    char hex[17];
    for (int i = 0; i < 8; i++) {
        sprintf(&hex[i*2], "%02x", fp->short_id[i]);
    }
    ESP_LOGI(TAG, "Model fingerprint: %s (size: %llu)", hex, (unsigned long long)fp->file_size);
}

void app_main(void) {
    ESP_LOGI(TAG, "Mallorn ESP32 Example");
    ESP_LOGI(TAG, "Buffer: %d bytes (min: %zu)", sizeof(mallorn_buffer), mallorn_min_buffer_size());

    ESP_ERROR_CHECK(nvs_flash_init());

    nvs_handle_t nvs;
    if (nvs_open("mallorn", NVS_READONLY, &nvs) == ESP_OK) {
        nvs_get_i32(nvs, "slot", &active_slot);
        nvs_close(nvs);
    }

    ESP_LOGI(TAG, "Active model: slot %c", 'A' + active_slot);

    // Get and log current model fingerprint (~10ms regardless of model size)
    mallorn_fingerprint_t fp;
    if (get_model_fingerprint(&fp) == ESP_OK) {
        log_fingerprint(&fp);

        // Example: Check if update is needed by comparing with server's expected version
        // uint8_t expected_short_id[8] = { 0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0x01, 0x02 };
        // if (!mallorn_fingerprint_check(&fp, expected_short_id)) {
        //     ESP_LOGI(TAG, "Update available!");
        //     apply_model_patch("http://server/patch.tflp", expected_hash);
        // }
    }

    // Example usage:
    // uint8_t hash[32] = {...};
    // apply_model_patch("http://server/patch.tflp", hash);

    while (1) vTaskDelay(pdMS_TO_TICKS(1000));
}
