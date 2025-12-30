#!/usr/bin/env python3
"""
List all Git worktrees with enhanced formatting.
"""
import subprocess
import sys
from pathlib import Path


def list_worktrees():
    """List all worktrees with formatted output."""
    try:
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
                current['branch'] = line.split('refs/heads/', 1)[1] if 'refs/heads/' in line else line.split(' ', 1)[1]
            elif line.startswith('HEAD '):
                current['commit'] = line.split(' ', 1)[1][:8]
            elif line.startswith('bare'):
                current['bare'] = True
            elif line.startswith('detached'):
                current['detached'] = True

        if current:
            worktrees.append(current)

        if not worktrees:
            print("No worktrees found")
            return

        print("\nGit Worktrees\n")
        print("=" * 80)

        for wt in worktrees:
            path = Path(wt['path'])
            is_main = '(main)' if path == Path.cwd() or 'bare' in wt else ''
            branch = wt.get('branch', wt.get('commit', 'unknown'))

            print(f"\nBranch: {branch} {is_main}")
            print(f"   Path: {path}")
            if 'commit' in wt:
                print(f"   Commit: {wt['commit']}")

        print("\n" + "=" * 80)
        print(f"Total: {len(worktrees)} worktree(s)\n")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    list_worktrees()
