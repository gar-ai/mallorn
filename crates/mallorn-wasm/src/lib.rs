//! Mallorn WebAssembly bindings
//!
//! Browser-compatible subset of Mallorn for in-browser model patching.
//!
//! # Limitations
//!
//! - No filesystem access (use ArrayBuffer inputs)
//! - Memory constraints (~500MB max practical)
//! - No network access (fetch patches externally)
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { apply_patch, fingerprint, create_patch } from 'mallorn_wasm';
//!
//! async function main() {
//!     await init();
//!
//!     // Load model and patch as ArrayBuffers
//!     const model = await fetch('model_v1.tflite').then(r => r.arrayBuffer());
//!     const patch = await fetch('update.tflp').then(r => r.arrayBuffer());
//!
//!     // Apply patch
//!     const newModel = apply_patch(new Uint8Array(model), new Uint8Array(patch));
//!
//!     // Get fingerprint
//!     const fp = fingerprint(new Uint8Array(model));
//!     console.log('Model version:', fp.short_id);
//! }
//! ```

use wasm_bindgen::prelude::*;

mod compress;
mod diff;
mod error;
mod fingerprint;
mod patch;

pub use compress::*;
pub use diff::*;
pub use error::*;
pub use fingerprint::*;
pub use patch::*;

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    // Set up panic hook for better error messages
    console_error_panic_hook::set_once();
}

/// Get the library version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }
}
