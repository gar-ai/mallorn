---
name: git-worktree-manager
description: Manage Git worktrees for parallel feature development. Use when the user wants to work on multiple branches simultaneously, create worktrees, switch between features, open terminals/editors for worktrees, or manage parallel development workflows.
---

# Git Worktree Manager

Automated Git worktree management with automatic terminal/editor opening.

## Available Scripts

### create_worktree.py
Creates a worktree and automatically opens it in Claude Code or VSCode.

```bash
python3 create_worktree.py --feature <name> [--editor claude|vscode|none]
```

### create_multiple_worktrees.py
Creates multiple worktrees at once and opens each in a separate terminal.

```bash
python3 create_multiple_worktrees.py feature1 feature2 feature3 [--editor claude|vscode|none]
```

### open_worktree.py
Opens existing worktrees in new terminals/editors.

```bash
python3 open_worktree.py <worktree-name>  # Open specific worktree
python3 open_worktree.py --all            # Open all worktrees
python3 open_worktree.py --all --editor vscode  # Open all in VSCode
```

### list_worktrees.py
Lists all worktrees with enhanced formatting.

```bash
python3 list_worktrees.py
```

### cleanup_worktrees.py
Interactive cleanup of completed worktrees.

```bash
python3 cleanup_worktrees.py
```

## Workflow Examples

**User says: "Set up worktrees for user-auth, payment-api, and dashboard"**
```bash
python3 create_multiple_worktrees.py user-auth payment-api dashboard
```
Result: Creates 3 worktrees, opens each in a new Claude Code terminal

**User says: "Create a worktree for bug fix and open it in VSCode"**
```bash
python3 create_worktree.py --feature bugfix --editor vscode
```

**User says: "Open all my worktrees"**
```bash
python3 open_worktree.py --all
```

**User says: "I want to work on API v2"**
```bash
python3 create_worktree.py --feature api-v2
```
Opens automatically in Claude Code

**User says: "Show me my current worktrees"**
```bash
python3 list_worktrees.py
```

**User says: "Clean up old worktrees"**
```bash
python3 cleanup_worktrees.py
```

## Platform Support

- **macOS**: Opens new Terminal.app windows with Claude Code
- **Linux**: Supports gnome-terminal, konsole, xterm
- **VSCode**: Works on all platforms (requires `code` command)

## Notes

- Each worktree gets its own terminal/editor window automatically
- Small delays between opening multiple terminals prevent stacking
- All worktrees share the same `.git` database
- Worktrees are created in the parent directory with format: `<repo-name>-<feature-name>`
- Requires: git, python3
