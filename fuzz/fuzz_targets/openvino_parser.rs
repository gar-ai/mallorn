#![no_main]

use libfuzzer_sys::fuzz_target;
use mallorn_openvino::OpenVINOParser;

fuzz_target!(|data: &[u8]| {
    // Fuzz OpenVINO XML parser
    // Should never panic, even with arbitrary input
    let parser = OpenVINOParser::new();
    let _ = parser.parse(data);
});
