//! Mallorn-Lite: Minimal C library for embedded model delta updates
//!
//! Provides a streaming patcher that works with just 1KB of RAM,
//! suitable for microcontrollers like ESP32 and STM32.
//!
//! # Features
//! - Caller-provided buffers (no malloc)
//! - Callback-based I/O (flash/UART/network agnostic)
//! - LZ4 decompression (lightweight)
//! - SHA256 verification
//!
//! # Example (C)
//! ```c
//! uint8_t buffer[1024];
//! mallorn_patcher_t patcher;
//!
//! mallorn_init(&patcher, buffer, sizeof(buffer));
//! mallorn_set_source(&patcher, read_flash, &source_ctx);
//! mallorn_set_patch(&patcher, read_flash, &patch_ctx);
//! mallorn_set_output(&patcher, write_flash, &output_ctx);
//!
//! while (mallorn_step(&patcher) == MALLORN_CONTINUE) {
//!     watchdog_reset();
//! }
//!
//! if (mallorn_verify(&patcher, expected_hash) == MALLORN_OK) {
//!     // Success!
//! }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

mod ffi;
mod format;
mod patcher;

pub use ffi::*;
pub use format::*;
pub use patcher::*;
