#!/usr/bin/env python3
"""
Open an existing worktree in Claude Code or VSCode.
"""
import subprocess
import sys
import argparse
import time
from pathlib import Path
import platform


def get_worktrees():
    """Get list of all worktrees."""
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
            current = {'path': Path(line.split(' ', 1)[1])}
        elif line.startswith('branch '):
            current['branch'] = line.split('refs/heads/', 1)[1] if 'refs/heads/' in line else None

    if current:
        worktrees.append(current)

    return worktrees


def open_in_editor(worktree_path, editor='claude'):
    """Open the worktree in the specified editor."""
    abs_path = Path(worktree_path).resolve()

    if not abs_path.exists():
        print(f"Worktree not found: {abs_path}")
        return False

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
            print(f"Opened Claude Code in new terminal for: {abs_path.name}")

        elif system == 'Linux':
            terminals = [
                ['gnome-terminal', '--', 'bash', '-c', f'cd {abs_path} && claude; exec bash'],
                ['konsole', '--workdir', str(abs_path), '-e', 'claude'],
                ['xterm', '-e', f'cd {abs_path} && claude'],
            ]

            for term_cmd in terminals:
                try:
                    subprocess.Popen(term_cmd)
                    print(f"Opened Claude Code in new terminal for: {abs_path.name}")
                    break
                except FileNotFoundError:
                    continue

    elif editor == 'vscode':
        try:
            subprocess.Popen(['code', str(abs_path)])
            print(f"Opened VSCode for: {abs_path.name}")
        except FileNotFoundError:
            print("VSCode not found. Install 'code' command")
            return False

    return True


def open_all_worktrees(editor='claude'):
    """Open all worktrees in separate windows."""
    worktrees = get_worktrees()

    main_path = Path(subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        capture_output=True, text=True, check=True
    ).stdout.strip())

    others = [wt for wt in worktrees if wt['path'] != main_path]

    if not others:
        print("No additional worktrees found")
        return

    print(f"\nOpening {len(others)} worktrees...\n")

    for wt in others:
        branch = wt.get('branch', 'unknown')
        print(f"Opening: {branch} at {wt['path'].name}")
        open_in_editor(wt['path'], editor)
        time.sleep(0.5)

    print(f"\nOpened {len(others)} worktree(s)")


def find_worktree(name):
    """Find a worktree by name or path."""
    worktrees = get_worktrees()

    # Try exact path match
    for wt in worktrees:
        if str(wt['path']) == name or wt['path'].name == name:
            return wt['path']

    # Try feature name match
    repo_name = Path(subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        capture_output=True, text=True, check=True
    ).stdout.strip()).name

    for wt in worktrees:
        if wt['path'].name == f"{repo_name}-{name}":
            return wt['path']

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Open worktrees in Claude Code or VSCode'
    )
    parser.add_argument(
        'worktree',
        nargs='?',
        help='Specific worktree to open (path or feature name)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Open all worktrees'
    )
    parser.add_argument(
        '--editor',
        choices=['claude', 'vscode'],
        default='claude',
        help='Editor to use (default: claude)'
    )

    args = parser.parse_args()

    if args.all:
        open_all_worktrees(args.editor)
    elif args.worktree:
        path = find_worktree(args.worktree)
        if path:
            open_in_editor(path, args.editor)
        else:
            print(f"Worktree not found: {args.worktree}")
            print("\nAvailable worktrees:")
            for wt in get_worktrees():
                print(f"  - {wt['path'].name}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
