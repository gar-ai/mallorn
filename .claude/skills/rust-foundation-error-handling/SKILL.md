---
name: rust-error-handling
description: Design error types using thiserror for libraries and anyhow for applications. Use when defining error hierarchies or handling fallible operations.
---

# Error Handling

Use thiserror for libraries (structured errors) and anyhow for applications (context-rich errors).

## Library Pattern: thiserror

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("S3 error: {0}")]
    S3(String),

    #[error("Media extraction error: {0}")]
    MediaExtraction(String),

    #[error("Python execution error: {0}")]
    Python(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

// Type alias for ergonomics
pub type Result<T> = std::result::Result<T, ProcessingError>;
```

## Key thiserror Features

```rust
#[derive(Error, Debug)]
pub enum MyError {
    // Automatic From implementation
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    // Transparent wrapper (delegates Display/source)
    #[error(transparent)]
    Other(#[from] anyhow::Error),

    // Formatted message with fields
    #[error("Failed to process {path}: {reason}")]
    ProcessFailed { path: String, reason: String },

    // Source error without From
    #[error("Database query failed")]
    Query(#[source] sqlx::Error),
}
```

## Application Pattern: anyhow

```rust
use anyhow::{Context, Result, bail, ensure};

fn load_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .context("Failed to read config file")?;

    let config: Config = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse config from {}", path))?;

    ensure!(config.is_valid(), "Invalid configuration");

    Ok(config)
}

fn process() -> Result<()> {
    if something_wrong {
        bail!("Something went wrong: {}", details);
    }
    Ok(())
}
```

## Error Propagation

```rust
// Use ? for automatic conversion
fn process_video(path: &Path) -> Result<Video> {
    let data = std::fs::read(path)?;           // io::Error -> ProcessingError
    let video = parse_video(&data)?;           // ParseError -> ProcessingError
    Ok(video)
}

// Add context when propagating
fn load_and_process(path: &Path) -> Result<Video> {
    let video = process_video(path)
        .map_err(|e| ProcessingError::MediaExtraction(
            format!("Failed to process {}: {}", path.display(), e)
        ))?;
    Ok(video)
}
```

## Domain-Specific Error Hierarchy

```rust
// Top-level error enum
#[derive(Error, Debug)]
pub enum AppError {
    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),

    #[error("API error: {0}")]
    Api(#[from] ApiError),
}

// Sub-domain errors
#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Connection failed: {0}")]
    Connection(String),

    #[error("Query failed: {0}")]
    Query(#[from] sqlx::Error),

    #[error("Record not found: {0}")]
    NotFound(String),
}

// Conversion between domains
impl From<DatabaseError> for ApiError {
    fn from(e: DatabaseError) -> Self {
        match e {
            DatabaseError::NotFound(id) => ApiError::NotFound(id),
            other => ApiError::Internal(other.to_string()),
        }
    }
}
```

## Retry-Aware Errors

```rust
#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error("Temporary error (retryable): {0}")]
    Temporary(String),

    #[error("Permanent error: {0}")]
    Permanent(String),
}

impl ProcessingError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, ProcessingError::Temporary(_))
    }
}
```

## Guidelines

- Use `thiserror` for library code (structured, typed errors)
- Use `anyhow` for application code (context-rich, flexible)
- Always derive `Debug` on error types
- Use `#[from]` for automatic conversion from common errors
- Add context with `.context()` or `.with_context()`
- Create domain-specific error variants
- Use `?` operator instead of `.unwrap()` or `.expect()`
- Consider if errors are retryable

## Examples

See `hercules-local-algo/src/error.rs` for production error hierarchy.
