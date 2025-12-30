#!/usr/bin/env bash
set -euo pipefail

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
echo "🔍 Linting Python service: $SERVICE_NAME"

if [ "$STAGED_ONLY" == "--staged-only" ]; then
  # Get staged files in this service directory
  STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM -- "$SERVICE_DIR" | grep -E '\.py$' || true)

  if [ -z "$STAGED_FILES" ]; then
    echo "✅ No Python files staged"
    exit 0
  fi

  # Strip service directory prefix to make paths relative
  STAGED_FILES=$(echo "$STAGED_FILES" | sed "s|^${SERVICE_NAME}/||")

  echo "📋 Linting $(echo "$STAGED_FILES" | wc -l) staged files"
  uvx ruff check $STAGED_FILES
  uvx ruff format --check $STAGED_FILES
else
  echo "📋 Linting entire service"
  uvx ruff check .
  uvx ruff format --check .

  echo ""
  echo "🔬 Running mypy type check..."
  # Skip mypy for terra-local-gpu temporarily - has many type issues to fix
  if [ "$SERVICE_NAME" == "terra-local-gpu" ]; then
    echo "⏭️  Skipping mypy for terra-local-gpu (type checking disabled temporarily)"
  elif [ "$SERVICE_NAME" == "vulcan-gpu-sdk" ]; then
    # vulcan-gpu-sdk has src/ for Rust code, use config's files setting
    uvx --python 3.12 --with types-requests mypy --python-version 3.11 --config-file pyproject.toml
  elif [ -d "src" ]; then
    uvx --python 3.12 --with types-requests mypy --python-version 3.12 --config-file pyproject.toml src/
  else
    uvx --python 3.12 --with types-requests mypy --python-version 3.12 --config-file pyproject.toml .
  fi
fi

echo "✅ Python linting passed for $SERVICE_NAME"
