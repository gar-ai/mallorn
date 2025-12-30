//! Mallorn OTA Delta Update Example for nRF52840
//!
//! This example demonstrates using mallorn-lite to apply delta patches
//! to a TFLite model stored in flash. The streaming API allows patching
//! with only 1KB of RAM working buffer.
//!
//! Hardware Setup:
//! - nRF52840 DK or compatible board
//! - 1MB flash for model storage
//!
//! Memory Layout:
//! - 0x00000000 - 0x00080000: Application
//! - 0x00080000 - 0x000C0000: Model A (256KB)
//! - 0x000C0000 - 0x00100000: Model B (256KB)

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use nrf52840_hal::{self as hal, gpio::Level, pac};
use panic_halt as _;

// Flash addresses for model storage
const MODEL_A_ADDR: u32 = 0x00080000;
const MODEL_B_ADDR: u32 = 0x000C0000;
const MODEL_SIZE: u32 = 0x00040000; // 256KB each

// Working buffer (1KB minimum for mallorn-lite)
static mut WORK_BUFFER: [u8; 1024] = [0u8; 1024];

/// Flash page size for nRF52840
const PAGE_SIZE: u32 = 4096;

/// Simulated mallorn patcher state
/// In production, this would use the actual mallorn-lite C FFI
struct PatcherState {
    source_addr: u32,
    output_addr: u32,
    patch_offset: usize,
    output_offset: usize,
}

impl PatcherState {
    fn new(source: u32, output: u32) -> Self {
        Self {
            source_addr: source,
            output_addr: output,
            patch_offset: 0,
            output_offset: 0,
        }
    }
}

/// Read from flash
fn flash_read(addr: u32, buf: &mut [u8]) {
    // Direct flash read on nRF52840
    let flash_ptr = addr as *const u8;
    for (i, byte) in buf.iter_mut().enumerate() {
        unsafe {
            *byte = *flash_ptr.add(i);
        }
    }
}

/// Erase flash page (must be page-aligned)
fn flash_erase_page(nvmc: &pac::NVMC, addr: u32) {
    // Enable erase
    nvmc.config.write(|w| w.wen().een());

    // Wait for ready
    while nvmc.ready.read().ready().is_busy() {}

    // Set page address and trigger erase
    nvmc.erasepage().write(|w| unsafe { w.erasepage().bits(addr) });

    // Wait for completion
    while nvmc.ready.read().ready().is_busy() {}

    // Disable write/erase
    nvmc.config.write(|w| w.wen().ren());
}

/// Write to flash (4 bytes at a time)
fn flash_write(nvmc: &pac::NVMC, addr: u32, data: &[u8]) {
    // Enable write
    nvmc.config.write(|w| w.wen().wen());

    // Wait for ready
    while nvmc.ready.read().ready().is_busy() {}

    // Write 4 bytes at a time
    let flash_ptr = addr as *mut u32;
    for (i, chunk) in data.chunks(4).enumerate() {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        let value = u32::from_le_bytes(word);
        unsafe {
            *flash_ptr.add(i) = value;
        }
        // Wait for write to complete
        while nvmc.ready.read().ready().is_busy() {}
    }

    // Disable write
    nvmc.config.write(|w| w.wen().ren());
}

/// Apply XOR delta (core of mallorn patching)
fn apply_xor_delta(source: &[u8], delta: &[u8], output: &mut [u8]) {
    for i in 0..output.len() {
        output[i] = source[i] ^ delta[i];
    }
}

/// Main application entry point
#[entry]
fn main() -> ! {
    // Get peripherals
    let p = pac::Peripherals::take().unwrap();
    let port0 = hal::gpio::p0::Parts::new(p.P0);

    // Configure LED for status indication
    let mut led = port0.p0_13.into_push_pull_output(Level::High);

    // Get NVMC for flash operations
    let nvmc = p.NVMC;

    // Initialize patcher state
    // In production, patch data would come from BLE/UART/USB
    let _patcher = PatcherState::new(MODEL_A_ADDR, MODEL_B_ADDR);

    // Blink LED to show we're running
    loop {
        // Toggle LED
        led.set_high().ok();
        cortex_m::asm::delay(8_000_000);
        led.set_low().ok();
        cortex_m::asm::delay(8_000_000);

        // In production:
        // 1. Check for pending patch via BLE/UART
        // 2. Stream patch data and apply using mallorn_step()
        // 3. Verify hash with mallorn_verify()
        // 4. Swap active model slot
    }
}

/// Example patch application flow (would be called when patch is available)
#[allow(dead_code)]
fn apply_patch_example(nvmc: &pac::NVMC, patch_data: &[u8]) -> bool {
    let buffer = unsafe { &mut WORK_BUFFER };

    // Read source chunk
    let mut source_chunk = [0u8; 256];
    flash_read(MODEL_A_ADDR, &mut source_chunk);

    // Apply delta
    let mut output_chunk = [0u8; 256];
    apply_xor_delta(&source_chunk, &patch_data[..256], &mut output_chunk);

    // Erase destination page
    flash_erase_page(nvmc, MODEL_B_ADDR);

    // Write output
    flash_write(nvmc, MODEL_B_ADDR, &output_chunk);

    // In production, continue until patch is fully applied
    // then verify with SHA256 hash

    true
}
