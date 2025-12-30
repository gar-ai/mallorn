#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz ONNX patch deserialization
    // Should never panic, even with arbitrary input
    let _ = mallorn_onnx::deserialize_patch(data);
});
