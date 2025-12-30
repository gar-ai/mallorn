/**
 * Mallorn STM32 Configuration
 */

#ifndef MALLORN_CONFIG_H
#define MALLORN_CONFIG_H

/* Model slot configuration */
#define MODEL_SLOT_A_ADDR  0x90000000
#define MODEL_SLOT_B_ADDR  0x90400000
#define MODEL_SLOT_SIZE    0x400000  /* 4MB */

/* Mallorn buffer size (minimum 1KB) */
#define MALLORN_BUFFER_SIZE 1024

#endif /* MALLORN_CONFIG_H */
