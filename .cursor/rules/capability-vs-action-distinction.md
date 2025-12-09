# Capability vs Action Distinction Rule

## Rule Overview
When users ask about capabilities or access, they are seeking information, NOT requesting actions to be performed.

## Critical Distinction

### Capability Questions (Informational Only)
- "Can you do X?"
- "Do you have access to X?"
- "Are you able to Y?"
- "Do you have the capability to Z?"

### Action Requests (Require Explicit Permission)
- "Do X"
- "Create Y"
- "Add Z"
- "Run this command"
- "Execute that function"

## Implementation Rules

1. **NEVER PERFORM ACTIONS** when user is only asking about capabilities
2. **ONLY PERFORM ACTIONS** when user explicitly requests them with clear action words
3. **RESPOND WITH INFORMATION ONLY** to capability questions
4. **WAIT FOR EXPLICIT ACTION REQUEST** before executing any functionality

## Examples

### ✅ Correct Response to Capability Question
**User**: "Can you add comments to Jira tasks?"
**Response**: "Yes, I have access to add comments to Jira tasks through the Atlassian MCP tool. I can add comments, create footer comments on Confluence pages, and create inline comments with text selection."

### ❌ Incorrect Response to Capability Question
**User**: "Can you add comments to Jira tasks?"
**Response**: *[Proceeds to add a comment to a Jira task]*

### ✅ Correct Response to Action Request
**User**: "Add a comment to AE-1159 saying the implementation is complete"
**Response**: *[Proceeds to add the requested comment]*

## Why This Rule Exists
- Prevents unwanted actions from being performed
- Respects user intent when they're only seeking information
- Maintains clear boundaries between information gathering and action execution
- Prevents accidental modifications to external systems

## Recent Violations
This rule has been violated multiple times in recent interactions:
- User asked "can I delete this PR?" - Agent offered to run deletion command instead of just providing information
- User asked "can you add comments to tasks?" - Agent immediately added a comment instead of just answering the capability question

## Enforcement
This rule is CRITICAL and must be followed in all interactions. Violating this rule can cause unwanted changes to external systems and repositories. The rule has been violated multiple times and requires strict adherence.
