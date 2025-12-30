/**
 * Flash I/O implementation for STM32F4
 *
 * This is a reference implementation. Adjust for your specific MCU.
 */

#include "flash_io.h"

#ifdef STM32F4
// STM32F4 flash registers (simplified)
#define FLASH_BASE         0x40023C00
#define FLASH_KEYR        (*(volatile uint32_t *)(FLASH_BASE + 0x04))
#define FLASH_SR          (*(volatile uint32_t *)(FLASH_BASE + 0x0C))
#define FLASH_CR          (*(volatile uint32_t *)(FLASH_BASE + 0x10))

#define FLASH_KEY1         0x45670123
#define FLASH_KEY2         0xCDEF89AB

#define FLASH_CR_PG        (1 << 0)
#define FLASH_CR_SER       (1 << 1)
#define FLASH_CR_STRT      (1 << 16)
#define FLASH_CR_LOCK      (1 << 31)
#define FLASH_SR_BSY       (1 << 16)

// Unlock flash for programming
static void flash_unlock(void)
{
    if (FLASH_CR & FLASH_CR_LOCK) {
        FLASH_KEYR = FLASH_KEY1;
        FLASH_KEYR = FLASH_KEY2;
    }
}

// Lock flash after programming
static void flash_lock(void)
{
    FLASH_CR |= FLASH_CR_LOCK;
}

// Wait for flash operation to complete
static void flash_wait(void)
{
    while (FLASH_SR & FLASH_SR_BSY) {
        // Optionally reset watchdog here
    }
}

// Get sector number from address (STM32F4 specific)
static int get_sector(uint32_t addr)
{
    // Simplified: assumes 128KB sectors starting at 0x08020000
    if (addr < 0x08020000) return -1;
    return (addr - 0x08000000) / (128 * 1024);
}

int flash_erase(uint32_t addr, size_t size)
{
    flash_unlock();

    uint32_t end_addr = addr + size;
    while (addr < end_addr) {
        int sector = get_sector(addr);
        if (sector < 0) {
            flash_lock();
            return -1;
        }

        flash_wait();

        // Erase sector
        FLASH_CR &= ~0xF8;  // Clear sector bits
        FLASH_CR |= (sector << 3) | FLASH_CR_SER;
        FLASH_CR |= FLASH_CR_STRT;

        flash_wait();

        FLASH_CR &= ~FLASH_CR_SER;

        addr += 128 * 1024;  // Next sector
    }

    flash_lock();
    return 0;
}

int flash_program(uint32_t addr, const uint8_t *data, size_t len)
{
    flash_unlock();

    FLASH_CR |= FLASH_CR_PG;

    // Program word by word (32-bit)
    for (size_t i = 0; i < len; i += 4) {
        flash_wait();

        uint32_t word = 0xFFFFFFFF;
        size_t remaining = len - i;
        if (remaining >= 4) {
            word = *(uint32_t *)(data + i);
        } else {
            // Partial word at end
            for (size_t j = 0; j < remaining; j++) {
                word &= ~(0xFF << (j * 8));
                word |= data[i + j] << (j * 8);
            }
        }

        *(volatile uint32_t *)(addr + i) = word;
    }

    flash_wait();
    FLASH_CR &= ~FLASH_CR_PG;
    flash_lock();

    return 0;
}

int flash_verify(uint32_t addr, const uint8_t *data, size_t len)
{
    const uint8_t *flash = (const uint8_t *)addr;
    for (size_t i = 0; i < len; i++) {
        if (flash[i] != data[i]) {
            return -1;
        }
    }
    return 0;
}

// Watchdog reset stub
void watchdog_reset(void)
{
    // IWDG->KR = 0xAAAA;  // Reload watchdog
}

#else
// Stub implementation for non-STM32 builds
int flash_erase(uint32_t addr, size_t size) { (void)addr; (void)size; return -1; }
int flash_program(uint32_t addr, const uint8_t *data, size_t len) { (void)addr; (void)data; (void)len; return -1; }
int flash_verify(uint32_t addr, const uint8_t *data, size_t len) { (void)addr; (void)data; (void)len; return -1; }
void watchdog_reset(void) {}
#endif
