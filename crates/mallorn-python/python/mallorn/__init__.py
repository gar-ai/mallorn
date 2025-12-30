"""
Mallorn - Edge model delta updates

Create minimal delta patches for ML models, reducing OTA bandwidth by 95%+.
Supports TFLite and GGUF model formats.

Example usage:
    >>> import mallorn
    >>> 
    >>> # Create a patch
    >>> stats = mallorn.create_patch("v1.tflite", "v2.tflite", "update.tflp")
    >>> print(f"Compression: {stats.compression_ratio:.1f}x")
    >>> 
    >>> # Apply the patch
    >>> result = mallorn.apply_patch("v1.tflite", "update.tflp", "v2.tflite")
    >>> assert result.is_valid()
"""

# Import from the native Rust module
from mallorn.mallorn import (
    create_patch,
    apply_patch,
    verify_patch,
    patch_info,
    PatchStats,
    PatchVerification,
    PatchInfo,
    DiffOptions,
)

__version__ = "0.1.0"

__all__ = [
    "create_patch",
    "apply_patch",
    "verify_patch",
    "patch_info",
    "PatchStats",
    "PatchVerification",
    "PatchInfo",
    "DiffOptions",
    "__version__",
]
