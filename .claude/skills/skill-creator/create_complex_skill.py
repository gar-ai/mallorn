#!/usr/bin/env python3
"""Create a complex skill with script templates."""

from __future__ import annotations

import argparse
import stat
import sys
from pathlib import Path


def get_skills_dir() -> Path:
    """Get the .claude/skills directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def generate_script_template(script_name: str, skill_name: str) -> str:
    """Generate a Python script template with argparse."""
    title = script_name.replace("_", " ").replace("-", " ").title()
    return f'''#!/usr/bin/env python3
"""{title} for {skill_name}."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="{title}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Add your arguments here
    # parser.add_argument("--example", "-e", help="Example argument")

    args = parser.parse_args()

    # TODO: Implement {script_name} logic
    print(f"Running {script_name}...")
    print("Not yet implemented")


if __name__ == "__main__":
    main()
'''


def generate_skill_md(name: str, description: str, scripts: list[str]) -> str:
    """Generate SKILL.md content for a complex skill."""
    title = name.replace("-", " ").title()

    # Generate script documentation
    script_docs = []
    for script in scripts:
        script_title = script.replace("_", " ").replace("-", " ").title()
        script_docs.append(f"""### {script}.py
{script_title}.

```bash
python3 {script}.py [options]
```""")

    scripts_section = "\n\n".join(script_docs)

    # Generate workflow examples
    workflow_examples = []
    for script in scripts[:3]:  # First 3 scripts for examples
        workflow_examples.append(f"""**User says: "Run {script}"**
```bash
python3 {script}.py
```""")

    workflows_section = "\n\n".join(workflow_examples)

    return f"""---
name: {name}
description: {description}
---

# {title}

[Describe the purpose and capabilities of this skill]

## Available Scripts

{scripts_section}

## Workflow Examples

{workflows_section}

## Notes

- [Add implementation notes]
- [Add dependencies or requirements]
- [Add platform-specific information if applicable]
"""


def create_complex_skill(name: str, description: str, scripts: list[str]) -> Path:
    """Create a complex skill with SKILL.md and script templates."""
    skills_dir = get_skills_dir()
    skill_dir = skills_dir / name

    if skill_dir.exists():
        print(f"Error: Skill '{name}' already exists at {skill_dir}")
        sys.exit(1)

    skill_dir.mkdir(parents=True)

    # Create SKILL.md
    skill_md_content = generate_skill_md(name, description, scripts)
    skill_md_path = skill_dir / "SKILL.md"
    skill_md_path.write_text(skill_md_content)

    # Create script templates
    for script in scripts:
        script_content = generate_script_template(script, name)
        script_path = skill_dir / f"{script}.py"
        script_path.write_text(script_content)

        # Make executable
        current_mode = script_path.stat().st_mode
        script_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return skill_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create a complex skill with script templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 create_complex_skill.py --name "deploy-manager" --description "Automate deployments" --scripts "deploy,rollback,status"
  python3 create_complex_skill.py -n "test-runner" -d "Run tests" -s "run,watch,coverage"
        """
    )
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Skill name (lowercase with hyphens)"
    )
    parser.add_argument(
        "--description", "-d",
        required=True,
        help="Skill description"
    )
    parser.add_argument(
        "--scripts", "-s",
        required=True,
        help="Comma-separated list of scripts to create (e.g., 'run,list,cleanup')"
    )

    args = parser.parse_args()

    # Parse scripts
    scripts = [s.strip().lower().replace(" ", "_") for s in args.scripts.split(",")]

    # Validate name
    name = args.name.lower().replace(" ", "-").replace("_", "-")

    skill_dir = create_complex_skill(name, args.description, scripts)

    print(f"""
Complex skill created successfully!
Path: {skill_dir}

Files created:
   SKILL.md""")
    for script in scripts:
        print(f"   {script}.py")

    print(f"""
Next steps:
   1. Edit {skill_dir}/SKILL.md to document your skill
   2. Implement logic in each script
   3. The skill will be auto-discovered by Claude Code
   4. Invoke with: skill: "{name}"
""")


if __name__ == "__main__":
    main()
