#!/bin/bash
# Quick switcher for worktrees

if [ $# -eq 0 ]; then
    echo "Usage: $0 <worktree-name-or-path>"
    exit 1
fi

WORKTREE=$1

# Check if it's a full path or just a name
if [ -d "$WORKTREE" ]; then
    cd "$WORKTREE" || exit 1
else
    # Try to find it in parent directory
    PARENT_DIR=$(git rev-parse --show-toplevel)/..
    REPO_NAME=$(basename "$(git rev-parse --show-toplevel)")
    FULL_PATH="$PARENT_DIR/${REPO_NAME}-${WORKTREE}"

    if [ -d "$FULL_PATH" ]; then
        cd "$FULL_PATH" || exit 1
    else
        echo "Worktree not found: $WORKTREE"
        exit 1
    fi
fi

echo "Switched to: $(pwd)"
echo "Branch: $(git branch --show-current)"

# Optionally start Claude Code
read -p "Start Claude Code here? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    claude
fi
