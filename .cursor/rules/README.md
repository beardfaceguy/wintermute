# Cursor Rules Overview

This directory contains rules and guidelines for AI assistant behavior when working on the Alix Estate Manager project.

## Rule Files Summary

### 1. `memory-tracking.md`
**Purpose**: Maintain cross-cutting insights and reusable knowledge
**Key Requirements**:
- Update `.cursor/memory/memory.md` with insights useful across multiple implementations
- Focus on architecture patterns, environment gotchas, and lessons learned
- Include critical commands, authentication info, and cross-cutting decisions
- Exclude task-specific progress (goes in IMPLEMENTATION_[Task].md files)

### 2. `implementation-tracking.md`
**Purpose**: Track detailed implementation progress for Jira tasks
**Key Requirements**:
- Create `IMPLEMENTATION_[Jira task name].md` files in `.cursor/docs/`
- Update both memory.md and implementation files together
- Document comprehensive progress, code changes, and decisions
- Maintain task-specific detailed documentation

### 3. `no-unescaped-exclamation.md`
**Purpose**: Prevent shell command issues with exclamation marks
**Key Requirements**:
- Never use unescaped `!` in shell commands
- Use `\!` or single quotes to escape exclamation marks
- Prevents bash history expansion errors

### 4. `interactive-commands.md`
**Purpose**: Handle interactive commands properly
**Key Requirements**:
- Use `--yes` or `--non-interactive` flags for non-interactive execution
- Avoid commands that require user input
- Use background mode for long-running processes

### 5. `package-manager-consistency.md`
**Purpose**: Ensure consistent package manager usage
**Key Requirements**:
- ALWAYS use yarn, NEVER use npm
- Maintain consistency across all package management operations
- Prevents CI/CD failures and deployment issues

### 6. `pre-commit-validation.md`
**Purpose**: Ensure code quality before commits
**Key Requirements**:
- Always run `yarn typecheck` and `yarn lint` before any commit
- Fix all TypeScript compilation and linting errors before committing
- Never commit code that fails validation checks
- Ensures CI/CD pipelines don't fail due to basic code issues

### 7. `requirement-clarification.md`
**Purpose**: Prevent unauthorized changes to requirements
**Key Requirements**:
- NEVER assume requirements or add "common practices" without explicit user approval
- STOP and ask for clarification when requirements are unclear or incomplete
- NEVER modify specifications without explicit approval
- Always ask first when in doubt about any implementation detail

### 8. `minimal-changes-principle.md`
**Purpose**: Prevent scope creep and architectural changes without approval
**Key Requirements**:
- ALWAYS GO FOR THE MINIMUM CHANGES NEEDED TO COMPLETE THE TASK
- Never make changes that affect more components than absolutely necessary
- Stop and get approval if task cannot be completed without major architectural changes
- Document all affected areas that would need testing for major changes
- Propose two-step approach for major changes: component change first, then feature addition

### 9. `deprecated-code-alert.md`
**Purpose**: Alert when working with deprecated code that needs modification
**Key Requirements**:
- ALWAYS ALERT WHEN WORKING WITH DEPRECATED CODE
- Stop and ask for team discussion or explicit approval before modifying deprecated code
- Document deprecated code and explain why change is needed
- Request developer explanation of why deprecated area is safe to work in
- Never assume deprecated code is safe to modify without approval

### 10. `capability-vs-action-distinction.md`
**Purpose**: Distinguish between capabilities and actions in AI responses
**Key Requirements**:
- Clearly separate what the AI can do vs what it will do
- Provide transparent capability descriptions
- Maintain clear action boundaries

### 11. `git-tagging-for-review.md`
**Purpose**: Use git tagging for code review processes
**Key Requirements**:
- Create appropriate git tags for review milestones
- Maintain clear version tracking for review purposes
- Follow established tagging conventions

## Usage Guidelines

### When Starting a New Session
1. Read `memory-tracking.md` to understand cross-cutting insights and patterns
2. Check for existing implementation files for current tasks
3. Review any pending work or issues from previous sessions

### During Development Work
1. Follow `implementation-tracking.md` for Jira task work
2. Apply `no-unescaped-exclamation.md` for shell commands
3. Use `interactive-commands.md` for automated processes
4. Apply `package-manager-consistency.md` for all package operations
5. **MANDATORY**: Apply `pre-commit-validation.md` before every commit
6. **CRITICAL**: Apply `requirement-clarification.md` before making any assumptions about requirements
7. **CRITICAL**: Apply `minimal-changes-principle.md` before making architectural changes
8. **CRITICAL**: Apply `deprecated-code-alert.md` when encountering deprecated code that needs modification
9. Apply `capability-vs-action-distinction.md` for clear AI response boundaries
10. Use `git-tagging-for-review.md` for review milestone tracking

### After Each Work Session
1. Update memory.md with any new cross-cutting insights or patterns discovered
2. Update relevant implementation files with progress
3. Document any new issues or discoveries in implementation files
4. Note next steps for future sessions in implementation files

## Rule Priority
1. **Safety First**: Always follow `no-unescaped-exclamation.md` and `interactive-commands.md`
2. **Requirements**: **CRITICAL** - Apply `requirement-clarification.md` before any assumptions
3. **Scope Control**: **CRITICAL** - Apply `minimal-changes-principle.md` before architectural changes
4. **Deprecated Code**: **CRITICAL** - Apply `deprecated-code-alert.md` when encountering deprecated code
5. **Code Quality**: **MANDATORY** - Apply `pre-commit-validation.md` before every commit
6. **Package Management**: **MANDATORY** - Apply `package-manager-consistency.md` for all package operations
7. **Documentation**: Maintain `memory-tracking.md` and `implementation-tracking.md`
8. **AI Boundaries**: Apply `capability-vs-action-distinction.md` for clear response boundaries
9. **Review Process**: Use `git-tagging-for-review.md` for milestone tracking

## File Locations
- **Rules**: `.cursor/rules/`
- **Memory**: `.cursor/memory/memory.md`
- **Implementation Docs**: `.cursor/docs/IMPLEMENTATION_[Task].md`
- **General Docs**: `.cursor/docs/`

## Maintenance
- Review and update rules as project evolves.  Note: **Any changes to the rules *HAVE TO* be approved by the user**
- Add new rules for recurring patterns or issues
- Ensure rules remain relevant and helpful
- Document any rule changes in this README
