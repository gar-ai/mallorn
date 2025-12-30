# Lint Monorepo Services

This skill runs linting across all services in the saturns-oracle monorepo.

## Services

The following services will be linted:

### TypeScript/JavaScript Services
- **apollos-ui**: NextJS UI application with ESLint and TypeScript

### Python Services
- **dark-magic-grimoire**: TikTok scraper with Ruff, Black, and mypy
- **model-orchestrator-service**: GPU job orchestration with Ruff, Black, and mypy
- **mercury-clustering**: Clustering service (basic Python linting)

## Tasks

Run linting for each service in the monorepo. For each service:

1. **apollos-ui (TypeScript/NextJS)**:
   - Run ESLint: `cd apollos-ui && pnpm lint`
   - Run TypeScript type checking: `cd apollos-ui && pnpm type-check`

2. **dark-magic-grimoire (Python)**:
   - Run Ruff linting: `cd dark-magic-grimoire && python -m ruff check .`
   - Run Black formatting check: `cd dark-magic-grimoire && python -m black --check .`
   - Run mypy type checking: `cd dark-magic-grimoire && python -m mypy .`

3. **model-orchestrator-service (Python)**:
   - Run Ruff linting: `cd model-orchestrator-service && python -m ruff check .`
   - Run Black formatting check: `cd model-orchestrator-service && python -m black --check .`
   - Run mypy type checking: `cd model-orchestrator-service && python -m mypy .`

4. **mercury-clustering (Python)**:
   - Run basic Python linting: `cd mercury-clustering && python -m py_compile *.py`

## Instructions

For each service:
1. Navigate to the service directory
2. Run all linting commands for that service
3. Collect any errors or warnings
4. Present a summary of:
   - Which services passed linting
   - Which services failed linting
   - Specific errors/warnings found
   - Suggested fixes

If there are linting errors:
- Clearly identify the file and line number
- Show the specific error message
- Suggest potential fixes if applicable

If all linting passes:
- Report success with a summary of what was checked

## Auto-fix Option

If the user wants to auto-fix linting issues:
- For apollos-ui: `cd apollos-ui && pnpm lint:fix`
- For Python services:
  - `cd <service> && python -m ruff check --fix .`
  - `cd <service> && python -m black .`

## Notes

- Run linting checks in parallel when possible for speed
- Always run from the repository root
- Use the appropriate package manager (pnpm for Node.js services)
- For Python services, ensure you're using the correct Python environment
- Skip mercury-clustering if it doesn't have Python files or if basic linting is not needed
