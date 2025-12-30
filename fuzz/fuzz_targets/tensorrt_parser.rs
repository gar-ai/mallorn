#![no_main]

use libfuzzer_sys::fuzz_target;
use mallorn_tensorrt::deserialize_patch;

fuzz_target!(|data: &[u8]| {
    // Fuzz TensorRT patch deserialization
    // TensorRT uses ONNX internally, but has its own patch format
    // Should never panic, even with arbitrary input
    let _ = deserialize_patch(data);
});
