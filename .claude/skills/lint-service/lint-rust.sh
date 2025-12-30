#!/usr/bin/env bash
set -euo pipefail

# Ensure Rust tools are in PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

SERVICE_NAME="${1:-}"
STAGED_ONLY="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
SERVICE_DIR="${REPO_ROOT}/${SERVICE_NAME}"

if [ -z "$SERVICE_NAME" ]; then
  echo "❌ Error: Service name required"
  echo "Usage: $0 <service-name> [--staged-only]"
  exit 1
fi

if [ ! -d "$SERVICE_DIR" ]; then
  echo "❌ Error: Service directory not found: $SERVICE_DIR"
  exit 1
fi

cd "$SERVICE_DIR"
echo "🔍 Linting Rust service: $SERVICE_NAME"

if [ "$STAGED_ONLY" == "--staged-only" ]; then
  # Get staged files in this service directory
  STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM -- "$SERVICE_DIR" | grep -E '\.rs$' || true)

  if [ -z "$STAGED_FILES" ]; then
    echo "✅ No Rust files staged"
    exit 0
  fi

  # Strip service directory prefix (for consistency, even though not strictly needed for Rust)
  STAGED_FILES=$(echo "$STAGED_FILES" | sed "s|^${SERVICE_NAME}/||")

  echo "📋 Rust files changed, running full lint (clippy requires it)"
fi

echo "🦀 Running clippy..."
cargo clippy --lib --bins --tests -- -D warnings

echo ""
echo "📐 Checking rustfmt..."
cargo fmt --all -- --check

if [ "$STAGED_ONLY" != "--staged-only" ]; then
  echo ""
  echo "🧪 Running tests..."
  cargo test --lib --bins
fi

echo "✅ Rust linting passed for $SERVICE_NAME"
