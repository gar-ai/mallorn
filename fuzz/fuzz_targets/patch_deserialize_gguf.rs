#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz GGUF patch deserialization
    // Should never panic, even with arbitrary input
    let _ = mallorn_gguf::deserialize_patch(data);
});
