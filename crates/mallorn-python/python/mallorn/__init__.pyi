"""
Mallorn - Edge model delta updates

Create minimal delta patches for ML models, reducing OTA bandwidth by 95%+.
Supports TFLite and GGUF model formats.
"""

from typing import Optional

class PatchStats:
    """Statistics from patch creation."""
    
    source_size: int
    """Size of the source model in bytes."""
    
    target_size: int
    """Size of the target model in bytes."""
    
    patch_size: int
    """Size of the generated patch in bytes."""
    
    compression_ratio: float
    """Compression ratio (source_size / patch_size)."""
    
    tensors_modified: int
    """Number of tensors that were modified."""
    
    tensors_unchanged: int
    """Number of tensors that remained unchanged."""

class PatchVerification:
    """Result of patch verification."""
    
    source_valid: bool
    """Whether the source model matches the patch's expected source hash."""
    
    patch_valid: bool
    """Whether the patch structure is valid."""
    
    expected_target_hash: str
    """Expected SHA256 hash of the patched model (hex-encoded)."""
    
    actual_target_hash: Optional[str]
    """Actual SHA256 hash of the patched model, if computed (hex-encoded)."""
    
    stats: PatchStats
    """Statistics about the patch operation."""
    
    def is_valid(self) -> bool:
        """Check if the verification passed (source and patch both valid)."""
        ...

class PatchInfo:
    """Information about a patch file."""
    
    format: str
    """Model format: 'tflite' or 'gguf'."""
    
    version: int
    """Patch format version."""
    
    source_hash: str
    """SHA256 hash of the source model (hex-encoded)."""
    
    target_hash: str
    """SHA256 hash of the target model (hex-encoded)."""
    
    compression: str
    """Compression method used: 'none', 'lz4', 'zstd(level)', or 'neural'."""
    
    operation_count: int
    """Total number of operations in the patch."""
    
    tensors_modified: int
    """Number of tensors that were modified."""
    
    tensors_unchanged: int
    """Number of tensors that were copied unchanged."""

class DiffOptions:
    """Options for diff creation."""
    
    compression: str
    """Compression algorithm: 'zstd' or 'lz4'."""
    
    compression_level: int
    """Compression level (1-22 for zstd)."""
    
    neural_compression: bool
    """Whether to use neural-optimized compression."""
    
    def __init__(
        self,
        compression: str = "zstd",
        compression_level: int = 3,
        neural_compression: bool = False,
    ) -> None:
        """Create diff options.
        
        Args:
            compression: Compression algorithm ('zstd' or 'lz4').
            compression_level: Compression level (1-22 for zstd, default 3).
            neural_compression: Enable neural-optimized compression for float tensors.
        """
        ...

def create_patch(
    old_model: str,
    new_model: str,
    output: Optional[str] = None,
    compression_level: int = 3,
    neural: bool = False,
) -> PatchStats:
    """Create a delta patch between two models.
    
    Args:
        old_model: Path to the source/old model file (.tflite or .gguf).
        new_model: Path to the target/new model file.
        output: Path for output patch file. If None, auto-generates based on new_model name.
        compression_level: Zstd compression level (1-22, default 3).
        neural: Enable neural-optimized compression for better float compression.
    
    Returns:
        PatchStats with compression statistics.
    
    Raises:
        ValueError: If models have different formats or files cannot be read.
    
    Example:
        >>> stats = mallorn.create_patch("v1.tflite", "v2.tflite", "update.tflp")
        >>> print(f"Compression: {stats.compression_ratio:.1f}x")
        Compression: 45.2x
    """
    ...

def apply_patch(model: str, patch: str, output: str) -> PatchVerification:
    """Apply a patch to a model.
    
    Args:
        model: Path to the source model file.
        patch: Path to the patch file (.tflp or .ggup).
        output: Path for the output patched model.
    
    Returns:
        PatchVerification with verification results.
    
    Raises:
        ValueError: If the patch cannot be applied (hash mismatch, corrupted, etc).
    
    Example:
        >>> result = mallorn.apply_patch("v1.tflite", "update.tflp", "v2.tflite")
        >>> assert result.is_valid()
    """
    ...

def verify_patch(model: str, patch: str) -> PatchVerification:
    """Verify a patch can be applied without actually applying it.
    
    Args:
        model: Path to the source model file.
        patch: Path to the patch file (.tflp or .ggup).
    
    Returns:
        PatchVerification with verification results.
    
    Raises:
        ValueError: If files cannot be read or patch is malformed.
    
    Example:
        >>> result = mallorn.verify_patch("v1.tflite", "update.tflp")
        >>> if result.source_valid:
        ...     print("Patch can be applied to this model")
    """
    ...

def patch_info(patch_path: str) -> PatchInfo:
    """Get information about a patch file.
    
    Args:
        patch_path: Path to the patch file (.tflp or .ggup).
    
    Returns:
        PatchInfo with patch details.
    
    Raises:
        ValueError: If the file cannot be read or is not a valid patch.
    
    Example:
        >>> info = mallorn.patch_info("update.tflp")
        >>> print(f"Format: {info.format}, Operations: {info.operation_count}")
    """
    ...

__version__: str
"""Package version."""

__all__ = [
    "create_patch",
    "apply_patch", 
    "verify_patch",
    "patch_info",
    "PatchStats",
    "PatchVerification",
    "PatchInfo",
    "DiffOptions",
]
