#!/usr/bin/env python3
"""List all existing Claude Code skills."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional


def get_skills_dir() -> Path:
    """Get the .claude/skills directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def parse_skill_md(skill_md_path: Path) -> dict:
    """Parse SKILL.md frontmatter to extract name and description."""
    content = skill_md_path.read_text()

    # Extract YAML frontmatter
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not frontmatter_match:
        return {"name": None, "description": None}

    frontmatter = frontmatter_match.group(1)

    # Parse name and description
    name_match = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
    desc_match = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)

    return {
        "name": name_match.group(1).strip() if name_match else None,
        "description": desc_match.group(1).strip() if desc_match else None,
    }


def get_skill_info(skill_dir: Path) -> dict | None:
    """Get information about a skill directory."""
    # Look for SKILL.md (case-insensitive)
    skill_md = None
    for name in ["SKILL.md", "skill.md"]:
        candidate = skill_dir / name
        if candidate.exists():
            skill_md = candidate
            break

    if not skill_md:
        return None

    info = parse_skill_md(skill_md)

    # Find scripts
    scripts = sorted([
        f.stem for f in skill_dir.glob("*.py")
    ])

    # Also check for shell scripts
    scripts.extend(sorted([
        f.stem for f in skill_dir.glob("*.sh")
    ]))

    return {
        "dir_name": skill_dir.name,
        "name": info["name"] or skill_dir.name,
        "description": info["description"] or "(no description)",
        "type": "complex" if scripts else "simple",
        "scripts": scripts,
        "path": skill_dir,
    }


def list_skills(verbose: bool = False):
    """List all skills in the skills directory."""
    skills_dir = get_skills_dir()

    skills = []
    for item in sorted(skills_dir.iterdir()):
        if item.is_dir():
            info = get_skill_info(item)
            if info:
                skills.append(info)
        elif item.suffix == ".md" and item.name != "README.md":
            # Flat file skills (like lint.md)
            skills.append({
                "dir_name": item.stem,
                "name": item.stem,
                "description": "(flat file skill)",
                "type": "simple",
                "scripts": [],
                "path": item,
            })

    if not skills:
        print("No skills found.")
        return

    print("Claude Code Skills")
    print("=" * 80)
    print()

    # Group by type
    simple_skills = [s for s in skills if s["type"] == "simple"]
    complex_skills = [s for s in skills if s["type"] == "complex"]

    if simple_skills:
        print("Documentation Skills (simple)")
        print("-" * 40)
        for skill in simple_skills:
            print(f"  {skill['name']}")
            if verbose:
                print(f"      {skill['description'][:70]}...")
                print()
        print()

    if complex_skills:
        print("Automation Skills (with scripts)")
        print("-" * 40)
        for skill in complex_skills:
            print(f"  {skill['name']}")
            if verbose:
                print(f"      {skill['description'][:70]}...")
                print(f"      Scripts: {', '.join(skill['scripts'])}")
                print()
        print()

    print("=" * 80)
    print(f"Total: {len(skills)} skill(s) ({len(simple_skills)} simple, {len(complex_skills)} complex)")
    print()
    print("To invoke a skill: skill: \"<name>\"")


def main():
    parser = argparse.ArgumentParser(
        description="List all Claude Code skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show descriptions and scripts for each skill"
    )

    args = parser.parse_args()
    list_skills(verbose=args.verbose)


if __name__ == "__main__":
    main()
