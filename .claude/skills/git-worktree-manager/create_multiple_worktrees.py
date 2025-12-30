#!/usr/bin/env python3
"""
Create multiple worktrees at once and open each in a separate terminal.
"""
import subprocess
import sys
import argparse
import time
from pathlib import Path


def create_worktree_batch(features, base_branch=None, editor='claude'):
    """Create multiple worktrees and open them."""
    print(f"\nCreating {len(features)} worktrees...\n")
    print("=" * 80)

    created = []
    failed = []

    skill_dir = Path(__file__).parent

    for i, feature in enumerate(features, 1):
        print(f"\n[{i}/{len(features)}] Creating worktree for: {feature}")
        print("-" * 40)

        cmd = ['python3', str(skill_dir / 'create_worktree.py'), '--feature', feature, '--editor', editor]
        if base_branch:
            cmd.extend(['--base', base_branch])

        try:
            result = subprocess.run(cmd, check=True)
            created.append(feature)

            if editor != 'none':
                time.sleep(0.5)

        except subprocess.CalledProcessError:
            failed.append(feature)
            print(f"Failed to create worktree for {feature}")

    print("\n" + "=" * 80)
    print("\nSummary:")
    print(f"   Created: {len(created)}")
    print(f"   Failed: {len(failed)}")

    if created:
        print(f"\nSuccessfully created worktrees for:")
        for feature in created:
            print(f"   - {feature}")

    if failed:
        print(f"\nFailed to create worktrees for:")
        for feature in failed:
            print(f"   - {feature}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Create multiple Git worktrees at once'
    )
    parser.add_argument(
        'features',
        nargs='+',
        help='Feature names (e.g., user-auth payment-api dashboard)'
    )
    parser.add_argument(
        '--base',
        help='Base branch to branch from (default: auto-detect)'
    )
    parser.add_argument(
        '--editor',
        choices=['claude', 'vscode', 'none'],
        default='claude',
        help='Editor to open (default: claude)'
    )

    args = parser.parse_args()

    create_worktree_batch(args.features, args.base, args.editor)


if __name__ == '__main__':
    main()
