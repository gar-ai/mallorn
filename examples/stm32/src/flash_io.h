/**
 * Flash I/O abstraction for STM32
 *
 * Implement these functions for your specific STM32 variant.
 */

#ifndef FLASH_IO_H
#define FLASH_IO_H

#include <stdint.h>
#include <stddef.h>

/**
 * Erase flash region
 *
 * @param addr Start address (must be sector-aligned)
 * @param size Size in bytes (must be sector-aligned)
 * @return 0 on success, negative on error
 */
int flash_erase(uint32_t addr, size_t size);

/**
 * Program flash
 *
 * @param addr Destination address
 * @param data Data to write
 * @param len Length in bytes
 * @return 0 on success, negative on error
 */
int flash_program(uint32_t addr, const uint8_t *data, size_t len);

/**
 * Verify flash contents
 *
 * @param addr Address to verify
 * @param data Expected data
 * @param len Length in bytes
 * @return 0 if match, negative on mismatch
 */
int flash_verify(uint32_t addr, const uint8_t *data, size_t len);

#endif /* FLASH_IO_H */
