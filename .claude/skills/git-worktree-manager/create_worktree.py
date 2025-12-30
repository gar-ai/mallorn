#!/usr/bin/env python3
"""
Create a Git worktree with smart defaults and validation.
Opens Claude Code or VSCode automatically.
"""
import subprocess
import sys
import os
import argparse
from pathlib import Path
import platform


def get_repo_name():
    """Get the current repository name."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True, text=True, check=True
        )
        return Path(result.stdout.strip()).name
    except subprocess.CalledProcessError:
        print("Error: Not in a git repository")
        sys.exit(1)


def get_default_branch():
    """Get the default branch (main or master)."""
    try:
        result = subprocess.run(
            ['git', 'symbolic-ref', 'refs/remotes/origin/HEAD'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split('/')[-1]
    except subprocess.CalledProcessError:
        # Fallback: check if main or master exists
        for branch in ['main', 'master']:
            result = subprocess.run(
                ['git', 'branch', '--list', branch],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                return branch
        return 'main'


def branch_exists(branch_name):
    """Check if a branch exists."""
    result = subprocess.run(
        ['git', 'branch', '--list', branch_name],
        capture_output=True, text=True
    )
    return bool(result.stdout.strip())


def open_in_editor(worktree_path, editor='claude'):
    """Open the worktree in the specified editor."""
    abs_path = worktree_path.resolve()

    if editor == 'claude':
        system = platform.system()

        if system == 'Darwin':  # macOS
            script = f'''
            tell application "Terminal"
                do script "cd {abs_path} && claude"
                activate
            end tell
            '''
            subprocess.Popen(['osascript', '-e', script])

        elif system == 'Linux':
            terminals = [
                ['gnome-terminal', '--', 'bash', '-c', f'cd {abs_path} && claude; exec bash'],
                ['konsole', '--workdir', str(abs_path), '-e', 'claude'],
                ['xterm', '-e', f'cd {abs_path} && claude'],
            ]

            for term_cmd in terminals:
                try:
                    subprocess.Popen(term_cmd)
                    break
                except FileNotFoundError:
                    continue
        else:
            print(f"Auto-open not supported on {system}. Please manually cd to {abs_path} and run 'claude'")

    elif editor == 'vscode':
        try:
            subprocess.Popen(['code', str(abs_path)])
        except FileNotFoundError:
            print("VSCode not found. Install 'code' command or use --editor claude")


def create_worktree(feature_name, branch_name=None, base_branch=None, editor='claude'):
    """Create a worktree with the given feature name."""
    repo_name = get_repo_name()

    if not base_branch:
        base_branch = get_default_branch()

    if not branch_name:
        branch_name = f"feature/{feature_name}"

    worktree_path = Path('..') / f"{repo_name}-{feature_name}"

    if worktree_path.exists():
        print(f"Error: Path {worktree_path} already exists")
        return False

    try:
        if branch_exists(branch_name):
            print(f"Using existing branch: {branch_name}")
            subprocess.run(
                ['git', 'worktree', 'add', str(worktree_path), branch_name],
                check=True
            )
        else:
            print(f"Creating new branch: {branch_name}")
            subprocess.run(
                ['git', 'worktree', 'add', '-b', branch_name, str(worktree_path), base_branch],
                check=True
            )

        print(f"\nWorktree created successfully!")
        print(f"Path: {worktree_path.resolve()}")
        print(f"Branch: {branch_name}")

        if editor and editor != 'none':
            print(f"\nOpening in {editor}...")
            open_in_editor(worktree_path, editor)
        else:
            print(f"\nTo start working:")
            print(f"   cd {worktree_path}")
            print(f"   claude")

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error creating worktree: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Create a Git worktree for parallel development'
    )
    parser.add_argument(
        '--feature',
        required=True,
        help='Feature name (e.g., user-auth, api-v2)'
    )
    parser.add_argument(
        '--branch',
        help='Branch name (default: feature/<feature-name>)'
    )
    parser.add_argument(
        '--base',
        help='Base branch to branch from (default: auto-detect main/master)'
    )
    parser.add_argument(
        '--editor',
        choices=['claude', 'vscode', 'none'],
        default='claude',
        help='Editor to open (default: claude)'
    )

    args = parser.parse_args()

    create_worktree(args.feature, args.branch, args.base, args.editor)


if __name__ == '__main__':
    main()
