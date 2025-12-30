idf_component_register(
    SRCS "main.c" "model_update.c"
    INCLUDE_DIRS "."
    REQUIRES nvs_flash esp_http_client
)
