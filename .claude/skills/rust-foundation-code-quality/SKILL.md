---
name: rust-code-quality
description: Apply rustfmt and clippy linting standards for consistent, safe Rust code. Use before commits and in CI pipelines.
---

# Code Quality

Maintain consistent, safe Rust code with rustfmt and clippy.

## Rustfmt Configuration

Create `rustfmt.toml` in project root:

```toml
max_width = 100
edition = "2021"
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
reorder_imports = true
```

Run before every commit:
```bash
cargo fmt
cargo fmt --check  # CI mode (fails if not formatted)
```

## Clippy Configuration

Add to `lib.rs` or `main.rs`:

```rust
#![warn(clippy::all, clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![warn(missing_docs)]
```

Or in `Cargo.toml`:

```toml
[lints.clippy]
all = "warn"
pedantic = "warn"
unwrap_used = "deny"
expect_used = "deny"
```

Run clippy:
```bash
cargo clippy
cargo clippy -- -D warnings  # CI mode (fails on warnings)
```

## Recommended Clippy Lints

```rust
// Deny dangerous patterns
#![deny(clippy::unwrap_used)]        // Use ? or expect with context
#![deny(clippy::expect_used)]        // Use proper error handling
#![deny(clippy::panic)]              // No panics in library code
#![deny(clippy::todo)]               // No TODOs in production

// Warn on style issues
#![warn(clippy::pedantic)]           // Stricter lints
#![warn(clippy::nursery)]            // Experimental lints
#![warn(clippy::cargo)]              // Cargo.toml lints
#![warn(missing_docs)]               // Document public items
```

## CI Integration

GitHub Actions example:

```yaml
name: Rust CI
on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy

      - name: Format check
        run: cargo fmt --check

      - name: Clippy
        run: cargo clippy -- -D warnings

      - name: Tests
        run: cargo test
```

## Common Clippy Fixes

```rust
// Bad: clippy::unwrap_used
let value = some_option.unwrap();

// Good: Use ? operator
let value = some_option.ok_or(Error::MissingValue)?;

// Bad: clippy::clone_on_ref_ptr
let arc2 = arc1.clone();

// Good: Use Arc::clone for clarity
let arc2 = Arc::clone(&arc1);

// Bad: clippy::redundant_closure
items.iter().map(|x| process(x))

// Good: Pass function directly
items.iter().map(process)
```

## Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/sh
cargo fmt --check || exit 1
cargo clippy -- -D warnings || exit 1
```

## Guidelines

- Run `cargo fmt` before every commit
- Fix all clippy warnings before merging
- Use `#[allow(clippy::...)]` sparingly with justification
- Enable pedantic lints for library code
- Document all public items

## Examples

See `hercules-local-algo/src/lib.rs` for lint configuration.
