#![no_main]

use libfuzzer_sys::fuzz_target;
use mallorn_core::streaming::{TensorIndex, TensorLocation};

fuzz_target!(|data: &[u8]| {
    // Fuzz streaming patcher components
    // Should never panic, even with arbitrary input

    // Test TensorIndex with random tensor locations
    if data.len() >= 24 {
        let mut index = TensorIndex::new();

        // Parse arbitrary data as tensor locations
        let mut offset = 0;
        while offset + 24 <= data.len() {
            let name = format!("tensor_{}", offset);
            let tensor_offset = u64::from_le_bytes(
                data[offset..offset + 8].try_into().unwrap_or([0; 8])
            );
            let size = u64::from_le_bytes(
                data[offset + 8..offset + 16].try_into().unwrap_or([0; 8])
            );

            // Don't allow unreasonable sizes
            let size = size % (1024 * 1024 * 100); // Max 100MB

            let location = TensorLocation::new(name, tensor_offset, size);
            index.add(location);

            offset += 24;
        }

        // Test index operations
        let _ = index.len();
        let _ = index.is_empty();
        let _ = index.total_size();

        for name in index.names() {
            let _ = index.get(name);
        }
    }
});
