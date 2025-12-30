//! Error types for WASM module

use wasm_bindgen::prelude::*;

/// Error type for WASM operations
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmError {
    message: String,
}

#[wasm_bindgen]
impl WasmError {
    /// Get the error message
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

impl WasmError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl From<&str> for WasmError {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for WasmError {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for WasmError {}
