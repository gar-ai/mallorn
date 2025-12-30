# Mallorn-lite component for ESP-IDF
# Link against the pre-built static library

idf_component_register(
    INCLUDE_DIRS "include"
)

# Link against pre-built mallorn-lite static library
# Build with: cargo build -p mallorn-lite --release --target xtensa-esp32-espidf
target_link_libraries(${COMPONENT_LIB} INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/libmallorn_lite.a"
)
