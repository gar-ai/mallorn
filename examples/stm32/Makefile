# Mallorn OTA Delta Update Example for STM32
#
# This Makefile builds the mallorn OTA example for STM32F4 series.
# Requires ARM GCC toolchain and mallorn-lite static library.

# Toolchain
CC = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy
SIZE = arm-none-eabi-size

# Target MCU (adjust for your board)
MCU = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard

# Directories
SRC_DIR = src
BUILD_DIR = build
MALLORN_DIR = ../../crates/mallorn-lite

# Source files
SRCS = $(SRC_DIR)/main.c $(SRC_DIR)/startup.c $(SRC_DIR)/flash_io.c

# Include paths
INCLUDES = -I$(MALLORN_DIR)/include -I$(SRC_DIR)

# Libraries
LIBS = -L$(MALLORN_DIR)/../../target/thumbv7em-none-eabihf/release -lmallorn_lite

# Compiler flags
CFLAGS = $(MCU) -Os -Wall -Wextra -std=c11
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -DSTM32F4

# Linker flags
LDFLAGS = $(MCU) -specs=nosys.specs -specs=nano.specs
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -T$(SRC_DIR)/stm32f4.ld

# Output
TARGET = mallorn_ota

.PHONY: all clean

all: $(BUILD_DIR)/$(TARGET).bin

$(BUILD_DIR)/$(TARGET).elf: $(SRCS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LIBS) -o $@
	$(SIZE) $@

$(BUILD_DIR)/$(TARGET).bin: $(BUILD_DIR)/$(TARGET).elf
	$(OBJCOPY) -O binary $< $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)

# Build mallorn-lite for ARM target
mallorn:
	cd ../.. && cargo build -p mallorn-lite --release --target thumbv7em-none-eabihf

.PHONY: mallorn
