---
name: rust-ml-huggingface-porting
description: Port HuggingFace models to Candle with correct tensor naming. Use when debugging weight loading errors or matmul issues.
---

# Porting HuggingFace Models to Candle

Critical patterns for loading HuggingFace safetensors weights into Candle models.

## CRITICAL: Tensor Naming Mismatches

HuggingFace models have inconsistent naming conventions. **Always inspect the safetensors first:**

```bash
# List all tensor names in a safetensors file
python -c "from safetensors import safe_open; f = safe_open('model.safetensors', 'pt'); print('\n'.join(sorted(f.keys())))"
```

### Common Naming Patterns by Model Family

| Model Family | Pattern | Example |
|--------------|---------|---------|
| **VideoMAE** | `model.blocks.{i}.attn.qkv.weight` | OpenGVLab |
| **VideoMAE** | `videomae.encoder.layer.{i}.attention.qkv.weight` | HuggingFace |
| **DINOv3** | `blocks.{i}.attn.qkv.weight` | Meta/timm |
| **CLAP Audio** | `audio_model.audio_encoder.layers.{i}.blocks.{j}...` | HuggingFace |
| **CLAP Text** | `text_model.encoder.layer.{i}.attention.self.query.weight` | HuggingFace |
| **Qwen3** | `layers.{i}.self_attn.q_proj.weight` (NO `model.` prefix) | HuggingFace |
| **LLaMA** | `model.layers.{i}.self_attn.q_proj.weight` (HAS `model.` prefix) | HuggingFace |

### Runtime Detection Pattern

```rust
fn detect_weight_format(vb: &VarBuilder) -> WeightFormat {
    // Try multiple naming conventions
    if vb.contains_tensor("model.blocks.0.attn.qkv.weight") {
        WeightFormat::OpenGVLab  // model. prefix
    } else if vb.contains_tensor("blocks.0.attn.qkv.weight") {
        WeightFormat::Timm  // No prefix
    } else if vb.contains_tensor("videomae.encoder.layer.0.attention.qkv.weight") {
        WeightFormat::HuggingFace
    } else {
        WeightFormat::Unknown
    }
}
```

### VarBuilder Prefix Chaining

```rust
// Navigate nested model structure
let vb_audio = vb.pp("audio_model").pp("audio_encoder");
let vb_layer = vb_audio.pp("layers").pp(format!("{}", layer_idx));
let vb_block = vb_layer.pp("blocks").pp(format!("{}", block_idx));

// Component paths differ by model family:
// CLAP Swin: attention.self.query, attention.self.key, attention.output.dense
// Standard ViT: attn.qkv (fused), attn.proj
// LLM: self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj
```

## CRITICAL: Contiguous Tensor Errors

**Error:** `matmul is only supported for contiguous tensors`

**Root Cause:** Operations like `transpose()`, `reshape()`, `narrow()`, and scalar multiplication create non-contiguous views.

### The Fix Pattern

```rust
// WRONG - will crash on CUDA
let q = (q * self.scale)?;
let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
let out = attn.matmul(&v)?;

// CORRECT - always call .contiguous() before matmul
let q = (q * self.scale)?.contiguous()?;  // Scale creates view
let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
let out = attn.contiguous()?.matmul(&v.contiguous()?)?;  // BOTH operands!
```

### When to Call `.contiguous()`

| After This Operation | Call `.contiguous()`? |
|---------------------|----------------------|
| `tensor.transpose(...)` | YES |
| `tensor * scalar` | YES |
| `tensor.narrow(...)` | YES |
| `tensor.reshape(...)` | Usually safe |
| `softmax(...)` | Sometimes needed |
| Before `matmul()` | ALWAYS on both operands |

## CRITICAL: where_cond Type Errors

**Error:** `where conditions should be u8/u32/i64, expected: U32, got: F32`

**Root Cause:** `where_cond` requires boolean-like tensor, but attention masks are often f32.

```rust
// WRONG - padding_mask is f32
let result = padding_mask.where_cond(&zeros, &neg_inf)?;

// CORRECT - convert to bool first
let padding_mask_bool = padding_mask.ne(0.0)?;  // Creates u8 tensor
let result = padding_mask_bool.where_cond(&zeros, &neg_inf)?;
```

## CRITICAL: Shape Mismatch in Attention Masks

**Error:** `shape mismatch in matmul, lhs: [1, 1, 1, 16, 48, 48], rhs: [1, 16, 48, 128]`

**Root Cause:** Calling `unsqueeze()` multiple times on already-expanded tensors.

```rust
// WRONG - mask gets too many dimensions
let mask = create_causal_mask(seq_len)?;  // (seq, seq)
let mask = if let Some(padding_mask) = attention_mask {
    mask.broadcast_add(&padding_mask)?  // Now 4D
} else {
    mask
};
let mask = mask.unsqueeze(0)?.unsqueeze(0)?;  // Now 6D! WRONG

// CORRECT - expand causal mask BEFORE combining
let causal_mask = create_causal_mask(seq_len)?;  // (seq, seq)
let causal_mask = causal_mask.unsqueeze(0)?.unsqueeze(0)?;  // (1, 1, seq, seq)

let mask = if let Some(padding_mask) = attention_mask {
    // padding_mask is (batch, 1, 1, seq)
    causal_mask.broadcast_add(&padding_mask)?  // (batch, 1, seq, seq)
} else {
    causal_mask
};
// Don't unsqueeze again!
```

## Common Layer Naming Differences

### Attention Layers

```rust
// Standard ViT (fused QKV)
let qkv = linear(dim, dim * 3, vb.pp("qkv"))?;

// CLAP/RoBERTa (separate Q/K/V)
let query = linear(dim, dim, vb.pp("self").pp("query"))?;
let key = linear(dim, dim, vb.pp("self").pp("key"))?;
let value = linear(dim, dim, vb.pp("self").pp("value"))?;
let output = linear(dim, dim, vb.pp("output").pp("dense"))?;

// LLM style
let q_proj = linear(dim, dim, vb.pp("q_proj"))?;
let k_proj = linear(dim, dim, vb.pp("k_proj"))?;
let v_proj = linear(dim, dim, vb.pp("v_proj"))?;
let o_proj = linear(dim, dim, vb.pp("o_proj"))?;
```

### Normalization Layers

```rust
// Standard ViT: norm1, norm2
let norm1 = layer_norm(dim, vb.pp("norm1"))?;
let norm2 = layer_norm(dim, vb.pp("norm2"))?;

// CLAP Swin: layernorm_before, layernorm_after
let norm1 = layer_norm(dim, vb.pp("layernorm_before"))?;
let norm2 = layer_norm(dim, vb.pp("layernorm_after"))?;

// RoBERTa: attention.output.LayerNorm, output.LayerNorm
let attn_norm = layer_norm(dim, vb.pp("attention").pp("output").pp("LayerNorm"))?;
let output_norm = layer_norm(dim, vb.pp("output").pp("LayerNorm"))?;
```

### MLP Layers

```rust
// Standard ViT: mlp.fc1, mlp.fc2
let fc1 = linear(dim, hidden, vb.pp("mlp").pp("fc1"))?;
let fc2 = linear(hidden, dim, vb.pp("mlp").pp("fc2"))?;

// CLAP Swin: intermediate.dense, output.dense
let intermediate = linear(dim, hidden, vb.pp("intermediate").pp("dense"))?;
let output = linear(hidden, dim, vb.pp("output").pp("dense"))?;

// LLM (SwiGLU): gate_proj, up_proj, down_proj
let gate = linear(dim, hidden, vb.pp("gate_proj"))?;
let up = linear(dim, hidden, vb.pp("up_proj"))?;
let down = linear(hidden, dim, vb.pp("down_proj"))?;
```

## Projection Layer Patterns

### Two-Stage Projection (CLAP)

```rust
// HuggingFace CLAP uses linear1 + linear2
struct CLAPProjection {
    linear1: Linear,
    linear2: Linear,
}

impl CLAPProjection {
    fn new(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        Ok(Self {
            linear1: linear(in_dim, out_dim, vb.pp("linear1"))?,
            linear2: linear(out_dim, out_dim, vb.pp("linear2"))?,
        })
    }
}
```

### Biased vs No-Bias Fallback

```rust
// Some models have bias, some don't - try both
let proj = if let Ok(with_bias) = linear(in_dim, out_dim, vb.pp("proj")) {
    with_bias
} else {
    linear_no_bias(in_dim, out_dim, vb.pp("proj"))?
};
```

## Window Attention Padding (Swin Transformer)

When spatial dimensions < window_size, pad before partitioning:

```rust
fn forward(&self, xs: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let ws = self.window_size;

    // Pad if needed
    let pad_h = (ws - h % ws) % ws;
    let pad_w = (ws - w % ws) % ws;

    let (x, padded_h, padded_w) = if pad_h > 0 || pad_w > 0 {
        let x = pad_tensor(&x, pad_h, pad_w)?;
        (x, h + pad_h, w + pad_w)
    } else {
        (x, h, w)
    };

    // ... do window attention with padded_h, padded_w ...

    // Remove padding after
    if pad_h > 0 || pad_w > 0 {
        x = x.narrow(1, 0, h)?.narrow(2, 0, w)?;
    }
}
```

## Debugging Checklist

1. **"cannot find tensor X"**
   - Inspect safetensors to find actual tensor names
   - Check for `model.` prefix presence/absence
   - Check for nested prefixes like `audio_encoder`

2. **"matmul is only supported for contiguous tensors"**
   - Add `.contiguous()` after transpose
   - Add `.contiguous()` after scalar multiplication
   - Add `.contiguous()` to BOTH matmul operands

3. **"shape mismatch in matmul"**
   - Print tensor shapes at each step
   - Check unsqueeze/squeeze operations
   - Verify attention mask dimensions match

4. **"where conditions should be u8/u32/i64"**
   - Convert f32 mask to bool with `.ne(0.0)?`
   - Or cast with `.to_dtype(DType::U8)?`

## Examples

- See `hercules-local-algo/src/dinov3/` for standard ViT with Axial RoPE
- See `hercules-local-algo/src/clap/` for Swin Transformer + RoBERTa patterns
- See `hercules-local-algo/src/qwen3/` for LLM with RoPE and RMSNorm
- See `hercules-local-algo/src/videomae/` for spatiotemporal video transformer
