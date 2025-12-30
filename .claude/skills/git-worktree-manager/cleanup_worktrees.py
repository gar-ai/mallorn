#!/usr/bin/env python3
"""
Interactive cleanup of Git worktrees.
"""
import subprocess
import sys
from pathlib import Path


def get_worktrees():
    """Get list of worktrees."""
    result = subprocess.run(
        ['git', 'worktree', 'list', '--porcelain'],
        capture_output=True, text=True, check=True
    )

    worktrees = []
    current = {}

    for line in result.stdout.strip().split('\n'):
        if line.startswith('worktree '):
            if current:
                worktrees.append(current)
            current = {'path': line.split(' ', 1)[1]}
        elif line.startswith('branch '):
            current['branch'] = line.split('refs/heads/', 1)[1] if 'refs/heads/' in line else None

    if current:
        worktrees.append(current)

    return worktrees


def get_default_branch():
    """Get the default branch (main or master)."""
    try:
        result = subprocess.run(
            ['git', 'symbolic-ref', 'refs/remotes/origin/HEAD'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split('/')[-1]
    except subprocess.CalledProcessError:
        for branch in ['main', 'master']:
            result = subprocess.run(
                ['git', 'branch', '--list', branch],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                return branch
        return 'main'


def is_branch_merged(branch):
    """Check if a branch is merged into the default branch."""
    default_branch = get_default_branch()
    try:
        result = subprocess.run(
            ['git', 'branch', '--merged', default_branch],
            capture_output=True, text=True, check=True
        )
        return branch in result.stdout
    except subprocess.CalledProcessError:
        return False


def cleanup_worktrees():
    """Interactively cleanup worktrees."""
    worktrees = get_worktrees()

    main_path = Path(subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        capture_output=True, text=True, check=True
    ).stdout.strip())

    removable = [wt for wt in worktrees if Path(wt['path']) != main_path]

    if not removable:
        print("No removable worktrees found")
        return

    print("\nWorktree Cleanup\n")
    print("=" * 80)

    for wt in removable:
        branch = wt.get('branch', 'unknown')
        path = Path(wt['path'])
        merged = is_branch_merged(branch) if branch else False

        print(f"\nBranch: {branch}")
        print(f"   Path: {path}")
        if merged:
            print(f"   Status: Merged into {get_default_branch()}")

        response = input(f"\n   Remove this worktree? [y/N]: ").lower()

        if response == 'y':
            try:
                subprocess.run(['git', 'worktree', 'remove', str(path)], check=True)
                print(f"   Removed {path}")

                if merged and branch:
                    delete_branch = input(f"   Delete merged branch '{branch}'? [y/N]: ").lower()
                    if delete_branch == 'y':
                        subprocess.run(['git', 'branch', '-d', branch], check=True)
                        print(f"   Deleted branch {branch}")
            except subprocess.CalledProcessError as e:
                print(f"   Error: {e}")

    print("\n" + "=" * 80)
    print("Cleanup complete!\n")


if __name__ == '__main__':
    cleanup_worktrees()
