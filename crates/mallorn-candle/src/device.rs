//! GPU device abstraction

use candle_core::Device;

/// GPU device wrapper
#[derive(Clone)]
pub struct GpuDevice {
    inner: Device,
    name: String,
}

impl GpuDevice {
    /// Create a CPU device (fallback when no GPU available)
    pub fn cpu() -> Self {
        Self {
            inner: Device::Cpu,
            name: "CPU".to_string(),
        }
    }

    /// Create a CUDA device
    #[cfg(feature = "cuda")]
    pub fn cuda(ordinal: usize) -> Result<Self, crate::GpuError> {
        let device = Device::cuda_if_available(ordinal)
            .map_err(|e| crate::GpuError::DeviceError(e.to_string()))?;
        Ok(Self {
            inner: device,
            name: format!("CUDA:{}", ordinal),
        })
    }

    /// Create a Metal device (macOS)
    #[cfg(feature = "metal")]
    pub fn metal(ordinal: usize) -> Result<Self, crate::GpuError> {
        let device =
            Device::new_metal(ordinal).map_err(|e| crate::GpuError::DeviceError(e.to_string()))?;
        Ok(Self {
            inner: device,
            name: format!("Metal:{}", ordinal),
        })
    }

    /// Get the best available device (GPU if available, else CPU)
    pub fn best_available() -> Self {
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Self::cuda(0) {
                return device;
            }
        }

        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Self::metal(0) {
                return device;
            }
        }

        Self::cpu()
    }

    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        !matches!(self.inner, Device::Cpu)
    }

    /// Get device name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the underlying Candle device
    pub(crate) fn inner(&self) -> &Device {
        &self.inner
    }
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuDevice({})", self.name)
    }
}
