---
name: rust-candle-core
description: Build native Rust ML models with Candle framework. Use when implementing vision transformers, LLMs, or audio models with GPU acceleration.
---

# Candle ML Framework

Pure Rust ML framework for building and running neural networks with GPU acceleration.

## Setup

```toml
# Cargo.toml
[dependencies]
candle-core = "0.4"
candle-nn = "0.4"
candle-transformers = "0.4"
hf-hub = "0.3"
tokenizers = "0.15"
image = "0.25"              # For vision models
symphonia = { version = "0.5", features = ["all"] }  # For audio

[features]
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
```

## Device Initialization

```rust
use candle_core::Device;

fn init_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            tracing::info!("Using CUDA device");
            return Ok(device);
        }
    }
    tracing::info!("Using CPU device");
    Ok(Device::Cpu)
}
```

## Loading Models from HuggingFace Hub

```rust
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{Repo, RepoType};
use std::path::{Path, PathBuf};

fn load_model(model_id: &str, cache_dir: Option<&Path>, device: &Device) -> Result<VarBuilder> {
    let cache_path = cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("models/hf"));

    std::env::set_var("HF_HOME", &cache_path);

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_path)
        .build()?;

    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    // Download weights
    let weights_path = repo.get("model.safetensors")?;

    // Memory-map weights for efficiency
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
    };

    Ok(vb)
}
```

## Core Tensor Operations

```rust
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::softmax;

// Create tensor from vector
let data = vec![1.0f32, 2.0, 3.0, 4.0];
let tensor = Tensor::from_vec(data, (2, 2), &device)?;

// Reshape and transpose
let reshaped = tensor.reshape((1, 4))?;
let transposed = tensor.transpose(0, 1)?;

// Matrix multiplication
let result = a.matmul(&b)?;

// Softmax along dimension
let probs = softmax(&logits, D::Minus1)?;

// L2 normalization
fn l2_normalize(embeddings: &Tensor) -> Result<Tensor> {
    let norm = embeddings
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f64::MAX)?;
    embeddings.broadcast_div(&norm)
}

// Argmax for sampling
let next_token = probs
    .argmax(D::Minus1)?
    .to_dtype(DType::U32)?
    .to_vec1::<u32>()?[0];
```

## Vision Transformer Patterns

### Patch Embedding

```rust
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, VarBuilder};

struct PatchEmbed {
    proj: Conv2d,
    num_patches: usize,
}

impl PatchEmbed {
    fn new(vb: VarBuilder, in_channels: usize, embed_dim: usize, patch_size: usize) -> Result<Self> {
        let proj = conv2d(
            in_channels,
            embed_dim,
            patch_size,
            Conv2dConfig {
                stride: patch_size,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;
        Ok(Self { proj, num_patches: (224 / patch_size).pow(2) })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.proj.forward(xs)?;
        let (b, c, _h, _w) = xs.dims4()?;
        // (B, C, H, W) -> (B, num_patches, C)
        xs.reshape((b, c, self.num_patches))?.transpose(1, 2)
    }
}
```

### Multi-Head Attention

**CRITICAL:** Always call `.contiguous()` before `matmul()` - transposes and scalar ops create non-contiguous views!

```rust
use candle_nn::{linear, Linear, VarBuilder};

struct Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            qkv: linear(dim, dim * 3, vb.pp("qkv"))?,
            proj: linear(dim, dim, vb.pp("proj"))?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;

        // Compute Q, K, V
        let qkv = self.qkv.forward(xs)?;
        let qkv = qkv.reshape((b, n, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // (3, B, H, N, D)

        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        // Scaled dot-product attention
        // CRITICAL: .contiguous() after scale and transpose!
        let q = (q * self.scale)?.contiguous()?;
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        let attn = softmax(&attn, D::Minus1)?;

        // CRITICAL: .contiguous() on BOTH operands before matmul
        let out = attn.contiguous()?.matmul(&v.contiguous()?)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, n, c))?;

        self.proj.forward(&out)
    }
}
```

### Image Preprocessing

```rust
const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

fn load_image(path: &Path, device: &Device) -> Result<Tensor> {
    let img = image::open(path)?;
    let img = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();

    let (width, height) = (img.width() as usize, img.height() as usize);
    let data = img.into_raw();

    // Normalize: (H, W, C) -> (C, H, W) with ImageNet stats
    let mut normalized = vec![0.0f32; 3 * height * width];
    for c in 0..3 {
        for h in 0..height {
            for w in 0..width {
                let src_idx = (h * width + w) * 3 + c;
                let dst_idx = c * height * width + h * width + w;
                let pixel = data[src_idx] as f32 / 255.0;
                normalized[dst_idx] = (pixel - IMAGE_MEAN[c]) / IMAGE_STD[c];
            }
        }
    }

    Tensor::from_vec(normalized, (1, 3, height, width), device)
}
```

## LLM Patterns

### Rotary Position Embeddings (RoPE)

```rust
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        let theta = 10000.0f32;
        let half_dim = dim / 2;

        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / dim as f32))
            .collect();

        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();

        let mut cos_cache = vec![0.0f32; max_seq_len * half_dim];
        let mut sin_cache = vec![0.0f32; max_seq_len * half_dim];

        for (pos_idx, &pos) in positions.iter().enumerate() {
            for (freq_idx, &freq) in freqs.iter().enumerate() {
                let angle = pos * freq;
                cos_cache[pos_idx * half_dim + freq_idx] = angle.cos();
                sin_cache[pos_idx * half_dim + freq_idx] = angle.sin();
            }
        }

        Ok(Self {
            cos: Tensor::from_vec(cos_cache, (max_seq_len, half_dim), device)?,
            sin: Tensor::from_vec(sin_cache, (max_seq_len, half_dim), device)?,
        })
    }

    fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_, _, seq_len, dim) = x.dims4()?;
        let half = dim / 2;

        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;

        let cos = self.cos.narrow(0, start_pos, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;

        let rotated_x1 = x1.broadcast_mul(&cos)?.broadcast_sub(&x2.broadcast_mul(&sin)?)?;
        let rotated_x2 = x1.broadcast_mul(&sin)?.broadcast_add(&x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[rotated_x1, rotated_x2], D::Minus1)
    }
}
```

### Causal Attention Mask

```rust
fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)
}
```

### RMS Normalization

```rust
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        xs.broadcast_mul(&self.weight)
    }
}
```

## Audio Model Patterns

### Mel Spectrogram

```rust
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;

fn pcm_to_mel(samples: &[f32], filters: &[f32], device: &Device) -> Result<Tensor> {
    let n_frames = (samples.len() - N_FFT) / HOP_LENGTH + 1;

    // Hann window
    let hann_window: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos()))
        .collect();

    // STFT
    let fft_size = N_FFT / 2 + 1;
    let mut magnitudes = vec![0.0f32; n_frames * fft_size];

    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP_LENGTH;
        let windowed: Vec<f32> = samples[start..start + N_FFT]
            .iter()
            .zip(&hann_window)
            .map(|(s, w)| s * w)
            .collect();

        for k in 0..fft_size {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for (n, &sample) in windowed.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / N_FFT as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }
            magnitudes[frame_idx * fft_size + k] = real * real + imag * imag;
        }
    }

    // Apply mel filterbank and log scale
    let mut mel_spec = vec![0.0f32; n_frames * N_MELS];
    for frame in 0..n_frames {
        for mel in 0..N_MELS {
            let mut sum = 0.0f32;
            for k in 0..fft_size {
                sum += filters[mel * fft_size + k] * magnitudes[frame * fft_size + k];
            }
            mel_spec[frame * N_MELS + mel] = sum.max(1e-10).ln();
        }
    }

    Tensor::from_vec(mel_spec, (1, N_MELS, n_frames), device)
}
```

### Audio Loading with Symphonia

```rust
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;

fn load_audio(path: &Path) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let probed = symphonia::default::get_probe()
        .format(&Default::default(), mss, &FormatOptions::default(), &Default::default())?;

    let mut format = probed.format;
    let track = format.default_track().ok_or("No audio track")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())?;

    let track_id = track.id;
    let mut samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };
        if packet.track_id() != track_id { continue; }

        let decoded = decoder.decode(&packet)?;
        let spec = *decoded.spec();
        let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);

        // Convert to mono
        let channel_samples = sample_buf.samples();
        if spec.channels.count() > 1 {
            let channels = spec.channels.count();
            for chunk in channel_samples.chunks(channels) {
                samples.push(chunk.iter().sum::<f32>() / channels as f32);
            }
        } else {
            samples.extend_from_slice(channel_samples);
        }
    }

    Ok(samples)
}
```

## VRAM Estimation

```rust
fn estimate_vram_gb(
    hidden_size: usize,
    num_layers: usize,
    vocab_size: usize,
    intermediate_size: usize,
) -> f32 {
    let embedding_params = vocab_size * hidden_size;
    let attention_params = num_layers * 4 * hidden_size * hidden_size;
    let mlp_params = num_layers * 3 * hidden_size * intermediate_size;
    let norm_params = num_layers * hidden_size * 2;

    let total = embedding_params + attention_params + mlp_params + norm_params;

    // Float32 = 4 bytes, plus 20% activation overhead
    (total as f32 * 4.0 * 1.2) / (1024.0 * 1024.0 * 1024.0)
}
```

## Global Model Caching

```rust
use std::sync::OnceLock;
use parking_lot::Mutex;

static MODEL: OnceLock<Mutex<MyModel>> = OnceLock::new();

pub fn get_model() -> &'static Mutex<MyModel> {
    MODEL.get_or_init(|| {
        tracing::info!("Loading model (first use)...");
        Mutex::new(MyModel::load_default().expect("Failed to load model"))
    })
}

pub fn preload_model() -> Result<()> {
    let _ = get_model();
    Ok(())
}
```

## Guidelines

- Use `cuda` feature for GPU acceleration
- Memory-map weights with `from_mmaped_safetensors` for efficient loading
- Cache models globally with `OnceLock` to avoid reloading
- Estimate VRAM before loading models to prevent OOM
- Use pre-norm transformer blocks (norm before attention/MLP)
- L2 normalize embeddings for similarity search
- Use tracing for observability in model loading/inference

## Examples

- Vision: `hercules-local-algo/src/dinov3/` - DINOv3 ViT implementation
- LLM: `hercules-local-algo/src/qwen3/` - Qwen3 decoder model
- Audio: `hercules-local-algo/src/clap/` - CLAP audio encoder
- Speech: `hercules-local-algo/src/whisper/` - Whisper transcription
