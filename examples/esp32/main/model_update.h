/**
 * Model update types for ESP32 mallorn example
 */

#ifndef MODEL_UPDATE_H
#define MODEL_UPDATE_H

#include "esp_partition.h"
#include "esp_http_client.h"

// Flash reader context
typedef struct {
    const esp_partition_t *partition;
    size_t offset;
    size_t size;
} flash_reader_t;

// Flash writer context
typedef struct {
    const esp_partition_t *partition;
    size_t offset;
} flash_writer_t;

// HTTP reader context
typedef struct {
    esp_http_client_handle_t client;
} http_reader_t;

#endif // MODEL_UPDATE_H
