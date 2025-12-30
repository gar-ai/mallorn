#!/usr/bin/env python3
"""Create a simple documentation skill."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def get_skills_dir() -> Path:
    """Get the .claude/skills directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def create_skill(name: str, description: str, category: str | None = None) -> Path:
    """Create a simple skill with SKILL.md."""
    skills_dir = get_skills_dir()
    skill_dir = skills_dir / name

    if skill_dir.exists():
        print(f"Error: Skill '{name}' already exists at {skill_dir}")
        sys.exit(1)

    skill_dir.mkdir(parents=True)

    # Generate title from name
    title = name.replace("-", " ").title()
    if category:
        title = f"{category.upper()}: {title.replace(f'{category.title()} ', '')}"

    skill_md = f"""---
name: {name}
description: {description}
---

# {title}

## Overview

[Describe the purpose and scope of this skill]

## Patterns

### Pattern 1: [Name]

```
[Code example]
```

### Pattern 2: [Name]

```
[Code example]
```

## Guidelines

- [Guideline 1]
- [Guideline 2]
- [Guideline 3]

## Examples

See `[path/to/example]` for production usage.
"""

    skill_md_path = skill_dir / "SKILL.md"
    skill_md_path.write_text(skill_md)

    return skill_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create a simple documentation skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 create_skill.py --name "my-pattern" --description "Use when implementing X..."
  python3 create_skill.py --name "rust-caching" --description "Caching patterns" --category rust
        """
    )
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Skill name (lowercase with hyphens, e.g., 'my-new-skill')"
    )
    parser.add_argument(
        "--description", "-d",
        required=True,
        help="Skill description (include 'Use when...' for discoverability)"
    )
    parser.add_argument(
        "--category", "-c",
        help="Optional category prefix (e.g., rust, git, aws)"
    )

    args = parser.parse_args()

    # Validate name
    name = args.name.lower().replace(" ", "-").replace("_", "-")
    if args.category and not name.startswith(f"{args.category}-"):
        name = f"{args.category}-{name}"

    skill_dir = create_skill(name, args.description, args.category)

    print(f"""
Skill created successfully!
Path: {skill_dir}

Files created:
   SKILL.md

Next steps:
   1. Edit {skill_dir}/SKILL.md to add your documentation
   2. The skill will be auto-discovered by Claude Code
   3. Invoke with: skill: "{name}"
""")


if __name__ == "__main__":
    main()
