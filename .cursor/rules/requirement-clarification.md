# Rule: Requirement Clarification Before Implementation

## Purpose
To prevent unauthorized changes to requirements and ensure all implementations match exactly what was specified, without assumptions or additions.

## Key Requirements
- **NEVER assume requirements**: If requirements are unclear or incomplete, STOP and ask for clarification
- **NEVER add "common practices"**: Don't add features, constraints, or behaviors not explicitly specified
- **NEVER modify specifications**: Don't change, enhance, or "improve" requirements without explicit approval
- **ALWAYS ask first**: When in doubt about any aspect of implementation, ask the user before proceeding

## When to Stop and Ask
- Requirements mention "alphanumeric" but don't specify character set
- Requirements are vague or could be interpreted multiple ways
- You want to add "best practices" or "common conventions"
- You're considering excluding characters, adding validation, or changing formats
- Any implementation detail not explicitly specified in requirements

## Examples of Oversteps to Avoid
- ❌ **BAD**: "I'll exclude confusing characters like I, O, 0, 1 for clarity"
- ❌ **BAD**: "I'll add format validation since it's a good practice"
- ❌ **BAD**: "I'll use a more secure random generator"
- ✅ **GOOD**: "The requirement says 'alphanumeric' - should I use all A-Z, 0-9 or exclude any characters?"

## Enforcement
This rule is **CRITICAL** and applies to ALL implementations. No exceptions for "minor improvements" or "common practices".

## Rationale
- **Prevents Scope Creep**: Keeps implementations focused on exact requirements
- **Maintains User Control**: Ensures user makes all design decisions
- **Avoids Rework**: Prevents implementing features that weren't requested
- **Respects Specifications**: Honors the original requirements as written
