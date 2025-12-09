# Rule: Pre-Commit Validation

## Purpose
To ensure all code changes pass TypeScript compilation and linting checks before committing, preventing broken code from entering the repository.

## Key Requirements
- **Always Run Checks**: Before any `git commit`, MUST run both `yarn typecheck` and `yarn lint`
- **Fix All Issues**: If either command fails, MUST fix all errors before proceeding with commit
- **No Exceptions**: Never commit code that fails TypeScript compilation or linting
- **Verify Success**: Both commands must exit with code 0 (success) before committing

## Commands to Run
```bash
# Check TypeScript compilation
yarn typecheck

# Check linting/formatting
yarn lint

# Only commit if both pass
git commit -m "Your commit message"
```

## Error Handling
- **TypeScript Errors**: Fix type annotations, imports, and syntax issues
- **Linting Errors**: Fix formatting, style, and code quality issues
- **Auto-fix**: Use `yarn lint --fix` when possible for automatic formatting fixes
- **Manual Fixes**: For complex issues, manually edit files to resolve errors

## Rationale
- **Prevents Broken Builds**: Ensures CI/CD pipelines don't fail due to compilation errors
- **Maintains Code Quality**: Enforces consistent coding standards across the project
- **Reduces Review Cycles**: Minimizes back-and-forth in PR reviews for basic issues
- **Team Productivity**: Prevents other developers from encountering broken code

## Example Workflow
```bash
# Make changes to files
# ... edit files ...

# Validate before commit
yarn typecheck  # Must pass
yarn lint       # Must pass

# Only commit if validation passes
git add .
git commit -m "Your changes"
```

## Enforcement
This rule is **MANDATORY** and applies to ALL commits. No exceptions for "quick fixes" or "minor changes".
