#![no_main]

use libfuzzer_sys::fuzz_target;
use mallorn_safetensors::SafeTensorsParser;

fuzz_target!(|data: &[u8]| {
    // Fuzz SafeTensors parser
    // Should never panic, even with arbitrary input
    let parser = SafeTensorsParser::new();
    let _ = parser.parse(data);
});
