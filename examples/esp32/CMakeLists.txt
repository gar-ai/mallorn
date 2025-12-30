# Mallorn ESP32 Example
# ESP-IDF project for demonstrating mallorn-lite OTA model updates

cmake_minimum_required(VERSION 3.16)

# Set the components directory
set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/components"
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(mallorn_esp32_example)
