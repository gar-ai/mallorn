#![no_main]

use libfuzzer_sys::fuzz_target;
use mallorn_coreml::CoreMLParser;

fuzz_target!(|data: &[u8]| {
    // Fuzz CoreML parser
    // Should never panic, even with arbitrary input
    let parser = CoreMLParser::new();
    let _ = parser.parse(data);
});
