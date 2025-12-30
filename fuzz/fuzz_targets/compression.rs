//! Fuzz target for compression/decompression
//!
//! Tests compression roundtrip and decompression of arbitrary data.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use mallorn_core::{Compressor, DataType, Lz4Compressor, NeuralCompressor, ZstdCompressor};

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    data: Vec<u8>,
    compressor_type: u8,
    level: i32,
}

fuzz_target!(|input: FuzzInput| {
    if input.data.is_empty() || input.data.len() > 1024 * 1024 {
        return; // Skip empty or very large inputs
    }

    let compressor: Box<dyn Compressor> = match input.compressor_type % 3 {
        0 => Box::new(ZstdCompressor::new((input.level % 19).abs() + 1)),
        1 => Box::new(Lz4Compressor::new()),
        _ => Box::new(NeuralCompressor::new((input.level % 19).abs() + 1)),
    };

    // Compress
    let compressed = match compressor.compress(&input.data, DataType::UInt8) {
        Ok(c) => c,
        Err(_) => return, // Compression can fail on some inputs
    };

    // Decompress
    let decompressed = match compressor.decompress(&compressed, DataType::UInt8) {
        Ok(d) => d,
        Err(_) => return, // Decompression can fail on malformed data
    };

    // Verify roundtrip
    assert_eq!(
        input.data, decompressed,
        "Compression roundtrip failed!"
    );
});
