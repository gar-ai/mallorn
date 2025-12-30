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
echo "🔍 Linting TypeScript service: $SERVICE_NAME"

if [ "$STAGED_ONLY" == "--staged-only" ]; then
  # Get staged files in this service directory only
  STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM -- "$SERVICE_DIR" | grep -E '\.(ts|tsx|js|jsx)$' || true)

  if [ -z "$STAGED_FILES" ]; then
    echo "✅ No TypeScript files staged"
    exit 0
  fi

  # Strip service directory prefix to make paths relative
  STAGED_FILES=$(echo "$STAGED_FILES" | sed "s|^${SERVICE_NAME}/||")

  echo "📋 Linting $(echo "$STAGED_FILES" | wc -l) staged files"
  echo "$STAGED_FILES" | xargs pnpm eslint --max-warnings=-1
else
  echo "📋 Linting entire service"
  pnpm lint --max-warnings=-1
fi

echo ""
echo "📦 Running TypeScript type check..."
pnpm type-check

echo "✅ TypeScript linting passed for $SERVICE_NAME"
