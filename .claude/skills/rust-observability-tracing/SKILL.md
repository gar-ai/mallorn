---
name: rust-tracing
description: Instrument code with tracing spans and structured logging. Use for observability and performance analysis.
---

# Tracing

Structured logging and spans with the tracing ecosystem.

## Setup

```toml
# Cargo.toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

## Initialize Subscriber

```rust
use tracing_subscriber::{fmt, EnvFilter};

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

// With JSON output for production
fn init_tracing_json() {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
}
```

## Log Levels

```rust
use tracing::{trace, debug, info, warn, error};

trace!("Very detailed trace info");
debug!("Debug info for development");
info!("Normal operation info");
warn!("Warning: something might be wrong");
error!("Error: operation failed");

// Control with RUST_LOG environment variable
// RUST_LOG=debug cargo run
// RUST_LOG=my_crate=debug,other_crate=warn cargo run
```

## Structured Fields

```rust
use tracing::info;

// Key-value fields
info!(
    video_id = %video.id,
    status = ?video.status,  // Debug format
    count = 42,
    "Processing video"
);

// Field formatting:
// %value - Display format
// ?value - Debug format
// value = %expr - Named Display
// value = ?expr - Named Debug
// value - Direct value

info!(
    user_id = %user.id,           // Display
    request = ?request,            // Debug
    latency_ms = elapsed.as_millis() as u64,
    success = true,
    "Request completed"
);
```

## Instrument Functions

```rust
use tracing::instrument;

// Automatically create span for function
#[instrument]
fn process_item(item: &Item) -> Result<()> {
    // Span automatically includes function name and arguments
    info!("Processing started");
    // ...
    Ok(())
}

// Skip sensitive arguments
#[instrument(skip(password, db_pool))]
async fn authenticate(
    username: &str,
    password: &str,
    db_pool: &PgPool,
) -> Result<User> {
    // ...
}

// Add extra fields
#[instrument(fields(video_id = %video.video_id, status = %video.status))]
async fn process_video(&self, video: &VideoRecord) -> Result<()> {
    // ...
}

// Custom span name
#[instrument(name = "video_processing", skip(self))]
async fn process(&self, video: &Video) -> Result<()> {
    // ...
}

// Set level
#[instrument(level = "debug")]
fn debug_operation() {
    // ...
}
```

## Manual Spans

```rust
use tracing::{span, Level, Span};

fn process_batch(items: &[Item]) -> Result<()> {
    let span = span!(Level::INFO, "process_batch", count = items.len());
    let _enter = span.enter();

    for item in items {
        process_item(item)?;
    }

    Ok(())
}

// Async-friendly with .instrument()
use tracing::Instrument;

async fn async_operation() -> Result<()> {
    let span = span!(Level::INFO, "async_op");

    async {
        // This code runs within the span
        do_work().await
    }
    .instrument(span)
    .await
}
```

## Span Events

```rust
use tracing::{info_span, info};

let span = info_span!("request", method = "GET", path = "/api/data");
let _enter = span.enter();

info!("Request started");
// ... processing ...
info!(status = 200, "Request completed");
```

## Error Logging

```rust
use tracing::error;

match operation() {
    Ok(result) => info!("Success: {:?}", result),
    Err(e) => {
        error!(
            error = %e,
            error_debug = ?e,
            "Operation failed"
        );
    }
}

// With source chain
fn log_error_chain(e: &dyn std::error::Error) {
    error!(error = %e, "Error occurred");
    let mut source = e.source();
    while let Some(cause) = source {
        error!(cause = %cause, "Caused by");
        source = cause.source();
    }
}
```

## Tokio Console Integration

```toml
# Cargo.toml
[dependencies]
console-subscriber = "0.2"
```

```rust
fn main() {
    // Initialize console subscriber for tokio-console
    console_subscriber::init();

    // Or combine with regular logging
    use tracing_subscriber::prelude::*;

    tracing_subscriber::registry()
        .with(console_subscriber::spawn())
        .with(tracing_subscriber::fmt::layer())
        .init();
}
```

Run with: `tokio-console`

## Custom Layers

```rust
use tracing_subscriber::prelude::*;
use tracing_subscriber::{fmt, EnvFilter};

fn init_multi_layer() {
    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(true);

    let filter_layer = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();
}
```

## Timing with Spans

```rust
use tracing::{info_span, Instrument};
use std::time::Instant;

async fn timed_operation() -> Result<()> {
    let start = Instant::now();

    let result = async {
        // Operation
        do_work().await
    }
    .instrument(info_span!("timed_op"))
    .await;

    info!(
        duration_ms = start.elapsed().as_millis() as u64,
        "Operation completed"
    );

    result
}
```

## Guidelines

- Use `#[instrument]` for automatic span creation
- Skip sensitive data with `skip(password, token)`
- Use structured fields instead of string formatting
- Use appropriate log levels (error > warn > info > debug > trace)
- Include context (IDs, counts, durations) in spans
- Use `?` for Debug format, `%` for Display format
- Configure with `RUST_LOG` environment variable

## Examples

See `hercules-local-algo/src/pipeline/processor.rs` for instrumented async code.
