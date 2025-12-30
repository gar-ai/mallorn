//! Fuzz target for TFLite parser
//!
//! Tests parser robustness against malformed TFLite data.

#![no_main]

use libfuzzer_sys::fuzz_target;
use mallorn_tflite::TFLiteParser;

fuzz_target!(|data: &[u8]| {
    let parser = TFLiteParser::new();

    // Try to parse arbitrary data - should not panic
    let _ = parser.parse(data);
});
