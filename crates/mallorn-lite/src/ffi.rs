//! C FFI bindings for mallorn-lite
//!
//! These functions are exported with C linkage for use from C code.

use crate::patcher::{StreamingPatcher, MIN_BUFFER_SIZE};

/// Result codes for mallorn operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MallornResult {
    /// Operation completed successfully
    Ok = 0,
    /// More work to do, call step() again
    Continue = 1,
    /// Invalid patch format or data
    ErrorInvalidPatch = -1,
    /// Hash verification failed
    ErrorHashMismatch = -2,
    /// Buffer too small (need at least 1KB)
    ErrorBufferTooSmall = -3,
    /// I/O error (read/write callback failed)
    ErrorIo = -4,
    /// Operation was aborted
    ErrorAborted = -5,
}

impl From<i32> for MallornResult {
    fn from(code: i32) -> Self {
        match code {
            0 => MallornResult::Ok,
            1 => MallornResult::Continue,
            -1 => MallornResult::ErrorInvalidPatch,
            -2 => MallornResult::ErrorHashMismatch,
            -3 => MallornResult::ErrorBufferTooSmall,
            -4 => MallornResult::ErrorIo,
            -5 => MallornResult::ErrorAborted,
            _ => MallornResult::ErrorIo,
        }
    }
}

/// Size of the opaque patcher struct in bytes
pub const MALLORN_PATCHER_SIZE: usize = core::mem::size_of::<StreamingPatcher>();

/// Opaque patcher context
///
/// Allocate this on the stack or in static memory.
/// Size is fixed and known at compile time.
#[repr(C, align(8))]
pub struct MallornPatcher {
    /// Opaque internal storage - do not access directly
    _opaque: [u8; 256], // Fixed size large enough for StreamingPatcher
}

impl MallornPatcher {
    /// Create a new uninitialized patcher
    pub const fn new() -> Self {
        Self {
            _opaque: [0u8; 256],
        }
    }

    /// Get mutable reference to inner patcher (unsafe cast)
    fn inner_mut(&mut self) -> &mut StreamingPatcher {
        debug_assert!(core::mem::size_of::<StreamingPatcher>() <= 256);
        unsafe { &mut *(self._opaque.as_mut_ptr() as *mut StreamingPatcher) }
    }

    /// Get reference to inner patcher (unsafe cast)
    fn inner(&self) -> &StreamingPatcher {
        debug_assert!(core::mem::size_of::<StreamingPatcher>() <= 256);
        unsafe { &*(self._opaque.as_ptr() as *const StreamingPatcher) }
    }
}

impl Default for MallornPatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Read callback function type
///
/// Called to read data from source model or patch file.
///
/// # Arguments
/// * `ctx` - User-provided context pointer
/// * `buf` - Buffer to read data into
/// * `max_len` - Maximum number of bytes to read
///
/// # Returns
/// Number of bytes actually read (0 on EOF or error)
pub type MallornReadFn = extern "C" fn(ctx: *mut u8, buf: *mut u8, max_len: usize) -> usize;

/// Write callback function type
///
/// Called to write data to output model.
///
/// # Arguments
/// * `ctx` - User-provided context pointer
/// * `buf` - Buffer containing data to write
/// * `len` - Number of bytes to write
///
/// # Returns
/// Number of bytes actually written
pub type MallornWriteFn = extern "C" fn(ctx: *mut u8, buf: *const u8, len: usize) -> usize;

/// Initialize a patcher with a working buffer.
///
/// # Arguments
/// * `patcher` - Pointer to caller-allocated MallornPatcher
/// * `buffer` - Pointer to working buffer (minimum 1KB)
/// * `buffer_size` - Size of buffer in bytes
///
/// # Returns
/// * `MALLORN_OK` - Initialization succeeded
/// * `MALLORN_ERROR_BUFFER_TOO_SMALL` - Buffer is less than 1KB
///
/// # Safety
/// * `patcher` must be a valid, non-null pointer
/// * `buffer` must point to valid memory of at least `buffer_size` bytes
/// * Both pointers must remain valid until patching completes or is aborted
#[no_mangle]
pub unsafe extern "C" fn mallorn_init(
    patcher: *mut MallornPatcher,
    buffer: *mut u8,
    buffer_size: usize,
) -> MallornResult {
    if patcher.is_null() {
        return MallornResult::ErrorIo;
    }

    if buffer.is_null() || buffer_size < MIN_BUFFER_SIZE {
        return MallornResult::ErrorBufferTooSmall;
    }

    // Initialize the StreamingPatcher at the start of the opaque buffer
    let inner = &mut *((*patcher)._opaque.as_mut_ptr() as *mut StreamingPatcher);
    *inner = StreamingPatcher::new();
    if inner.init(buffer, buffer_size) {
        MallornResult::Ok
    } else {
        MallornResult::ErrorBufferTooSmall
    }
}

/// Set source model read callback.
///
/// # Arguments
/// * `patcher` - Pointer to initialized MallornPatcher
/// * `read_fn` - Callback function for reading source model data
/// * `ctx` - User context passed to callback
///
/// # Safety
/// * `patcher` must be a valid, initialized MallornPatcher
/// * `read_fn` must be a valid function pointer
/// * `ctx` must remain valid for the duration of patching
#[no_mangle]
pub unsafe extern "C" fn mallorn_set_source(
    patcher: *mut MallornPatcher,
    read_fn: MallornReadFn,
    ctx: *mut u8,
) {
    if patcher.is_null() {
        return;
    }

    (*patcher).inner_mut().set_source(read_fn, ctx);
}

/// Set patch file read callback.
///
/// # Arguments
/// * `patcher` - Pointer to initialized MallornPatcher
/// * `read_fn` - Callback function for reading patch data
/// * `ctx` - User context passed to callback
///
/// # Safety
/// * `patcher` must be a valid, initialized MallornPatcher
/// * `read_fn` must be a valid function pointer
/// * `ctx` must remain valid for the duration of patching
#[no_mangle]
pub unsafe extern "C" fn mallorn_set_patch(
    patcher: *mut MallornPatcher,
    read_fn: MallornReadFn,
    ctx: *mut u8,
) {
    if patcher.is_null() {
        return;
    }

    (*patcher).inner_mut().set_patch(read_fn, ctx);
}

/// Set output model write callback.
///
/// # Arguments
/// * `patcher` - Pointer to initialized MallornPatcher
/// * `write_fn` - Callback function for writing output data
/// * `ctx` - User context passed to callback
///
/// # Safety
/// * `patcher` must be a valid, initialized MallornPatcher
/// * `write_fn` must be a valid function pointer
/// * `ctx` must remain valid for the duration of patching
#[no_mangle]
pub unsafe extern "C" fn mallorn_set_output(
    patcher: *mut MallornPatcher,
    write_fn: MallornWriteFn,
    ctx: *mut u8,
) {
    if patcher.is_null() {
        return;
    }

    (*patcher).inner_mut().set_output(write_fn, ctx);
}

/// Execute one step of the patching process.
///
/// Call this in a loop until it returns something other than `MALLORN_CONTINUE`.
///
/// # Arguments
/// * `patcher` - Pointer to initialized MallornPatcher
///
/// # Returns
/// * `MALLORN_OK` - Patching completed successfully
/// * `MALLORN_CONTINUE` - More work to do, call again
/// * Negative value - Error occurred
///
/// # Example
/// ```c
/// while (mallorn_step(&patcher) == MALLORN_CONTINUE) {
///     watchdog_reset();  // Pet the watchdog between steps
/// }
/// ```
///
/// # Safety
/// * `patcher` must be a valid, initialized MallornPatcher
/// * I/O callbacks must have been set before first call
#[no_mangle]
pub unsafe extern "C" fn mallorn_step(patcher: *mut MallornPatcher) -> MallornResult {
    if patcher.is_null() {
        return MallornResult::ErrorIo;
    }

    let result = (*patcher).inner_mut().step();
    MallornResult::from(result)
}

/// Verify the patched output against expected hash.
///
/// Call this after `mallorn_step()` returns `MALLORN_OK`.
///
/// # Arguments
/// * `patcher` - Pointer to MallornPatcher that completed patching
/// * `expected_hash` - Pointer to 32-byte SHA256 hash to compare
///
/// # Returns
/// * `MALLORN_OK` - Hash matches
/// * `MALLORN_ERROR_HASH_MISMATCH` - Hash does not match
///
/// # Safety
/// * `patcher` must be a valid MallornPatcher
/// * `expected_hash` must point to at least 32 bytes
#[no_mangle]
pub unsafe extern "C" fn mallorn_verify(
    patcher: *const MallornPatcher,
    expected_hash: *const u8,
) -> MallornResult {
    if patcher.is_null() || expected_hash.is_null() {
        return MallornResult::ErrorIo;
    }

    let hash_slice: &[u8; 32] = &*(expected_hash as *const [u8; 32]);
    if (*patcher).inner().verify(hash_slice) {
        MallornResult::Ok
    } else {
        MallornResult::ErrorHashMismatch
    }
}

/// Abort the patching operation.
///
/// Safe to call at any time. After calling this, the patcher
/// cannot be reused without calling `mallorn_init()` again.
///
/// # Arguments
/// * `patcher` - Pointer to MallornPatcher
///
/// # Safety
/// * `patcher` may be null (no-op in that case)
#[no_mangle]
pub unsafe extern "C" fn mallorn_abort(patcher: *mut MallornPatcher) {
    if patcher.is_null() {
        return;
    }

    (*patcher).inner_mut().abort();
}

/// Get the minimum required buffer size.
///
/// # Returns
/// Minimum buffer size in bytes (1024)
#[no_mangle]
pub extern "C" fn mallorn_min_buffer_size() -> usize {
    MIN_BUFFER_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_conversion() {
        assert_eq!(MallornResult::from(0), MallornResult::Ok);
        assert_eq!(MallornResult::from(1), MallornResult::Continue);
        assert_eq!(MallornResult::from(-1), MallornResult::ErrorInvalidPatch);
        assert_eq!(MallornResult::from(-2), MallornResult::ErrorHashMismatch);
        assert_eq!(MallornResult::from(-3), MallornResult::ErrorBufferTooSmall);
        assert_eq!(MallornResult::from(-4), MallornResult::ErrorIo);
        assert_eq!(MallornResult::from(-5), MallornResult::ErrorAborted);
    }

    #[test]
    fn test_patcher_size() {
        // Ensure the patcher has a reasonable fixed size
        let size = core::mem::size_of::<MallornPatcher>();
        assert!(size < 512, "Patcher too large: {} bytes", size);
    }

    #[test]
    fn test_init_null_patcher() {
        let mut buffer = [0u8; 1024];
        let result =
            unsafe { mallorn_init(core::ptr::null_mut(), buffer.as_mut_ptr(), buffer.len()) };
        assert_eq!(result, MallornResult::ErrorIo);
    }

    #[test]
    fn test_init_null_buffer() {
        let mut patcher = MallornPatcher::new();
        let result = unsafe { mallorn_init(&mut patcher, core::ptr::null_mut(), 1024) };
        assert_eq!(result, MallornResult::ErrorBufferTooSmall);
    }

    #[test]
    fn test_init_small_buffer() {
        let mut patcher = MallornPatcher::new();
        let mut buffer = [0u8; 512];
        let result = unsafe { mallorn_init(&mut patcher, buffer.as_mut_ptr(), buffer.len()) };
        assert_eq!(result, MallornResult::ErrorBufferTooSmall);
    }

    #[test]
    fn test_init_success() {
        let mut patcher = MallornPatcher::new();
        let mut buffer = [0u8; 1024];
        let result = unsafe { mallorn_init(&mut patcher, buffer.as_mut_ptr(), buffer.len()) };
        assert_eq!(result, MallornResult::Ok);
    }
}
