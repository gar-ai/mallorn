//! Fuzz target for GGUF parser
//!
//! Tests parser robustness against malformed GGUF data.

#![no_main]

use libfuzzer_sys::fuzz_target;
use mallorn_gguf::GGUFParser;

fuzz_target!(|data: &[u8]| {
    let parser = GGUFParser::new();

    // Try to parse arbitrary data - should not panic
    let _ = parser.parse(data);
});
