//! Mallorn Python bindings
//!
//! Provides Python API for edge model delta updates.
//!
//! # Example
//! ```python
//! import mallorn
//!
//! # Create a patch
//! stats = mallorn.create_patch("v1.tflite", "v2.tflite", "update.tflp")
//! print(f"Compression ratio: {stats.compression_ratio:.1f}x")
//!
//! # Apply a patch
//! result = mallorn.apply_patch("v1.tflite", "update.tflp", "v2.tflite")
//! assert result.source_valid and result.patch_valid
//!
//! # Quick fingerprinting (~10ms for any model size)
//! fp = mallorn.fingerprint("model.tflite")
//! print(f"Model version: {fp.short()}")
//!
//! # Compare models by fingerprint
//! same = mallorn.compare_fingerprints("model_a.tflite", "model_b.tflite")
//! ```

use pyo3::prelude::*;

mod api;

/// Python module definition
#[pymodule]
fn mallorn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Functions
    m.add_function(wrap_pyfunction!(api::create_patch, m)?)?;
    m.add_function(wrap_pyfunction!(api::apply_patch, m)?)?;
    m.add_function(wrap_pyfunction!(api::verify_patch, m)?)?;
    m.add_function(wrap_pyfunction!(api::patch_info, m)?)?;
    m.add_function(wrap_pyfunction!(api::fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(api::compare_fingerprints, m)?)?;

    // Classes
    m.add_class::<api::PatchStats>()?;
    m.add_class::<api::PatchVerification>()?;
    m.add_class::<api::PatchInfo>()?;
    m.add_class::<api::DiffOptions>()?;
    m.add_class::<api::Fingerprint>()?;
    Ok(())
}
