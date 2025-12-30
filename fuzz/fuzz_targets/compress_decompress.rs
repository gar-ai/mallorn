#![no_main]

use libfuzzer_sys::fuzz_target;
use mallorn_core::{Compressor, DataType, Lz4Compressor, ZstdCompressor};

fuzz_target!(|data: &[u8]| {
    // Fuzz compression/decompression roundtrip
    // Should never panic and should always produce valid output

    // Test Zstd
    let zstd = ZstdCompressor::new(3);
    if let Ok(compressed) = zstd.compress(data, DataType::Float32) {
        let _ = zstd.decompress(&compressed, DataType::Float32);
    }

    // Test LZ4
    let lz4 = Lz4Compressor::new();
    if let Ok(compressed) = lz4.compress(data, DataType::Float32) {
        let _ = lz4.decompress(&compressed, DataType::Float32);
    }

    // Test decompression of arbitrary data (should not panic)
    let _ = zstd.decompress(data, DataType::Float32);
    let _ = lz4.decompress(data, DataType::Float32);
});
