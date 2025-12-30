#![no_main]

use libfuzzer_sys::fuzz_target;
use mallorn_core::{is_signed_patch, SignedPatch};

fuzz_target!(|data: &[u8]| {
    // Fuzz signed patch parsing
    // Should never panic, even with arbitrary input

    // Check magic detection
    let _ = is_signed_patch(data);

    // Try to parse as signed patch
    if let Ok(signed) = SignedPatch::from_bytes(data) {
        // If parsing succeeds, verification should not panic
        let _ = signed.verify();

        // Roundtrip should work
        let bytes = signed.to_bytes();
        let _ = SignedPatch::from_bytes(&bytes);
    }
});
