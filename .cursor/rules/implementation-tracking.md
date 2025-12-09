# Implementation File Tracking Rule

## Overview
When working on Jira tasks, always maintain both memory.md and a dedicated implementation file for comprehensive progress tracking.

## Rule Requirements

### 1. Implementation File Naming Convention
- **Format**: `IMPLEMENTATION_[Jira task name].md`
- **Location**: `.cursor/docs/`
- **Examples**:
  - `IMPLEMENTATION_AE-1159.md` (for task AE-1159)
  - `IMPLEMENTATION_AE-405.md` (for task AE-405)
  - `IMPLEMENTATION_AE-1107.md` (for task AE-1107)

### 2. When to Update Implementation Files
- **At the start** of any Jira task work
- **After each major milestone** or phase completion
- **When making significant code changes** or discoveries
- **When encountering issues** or blockers
- **At the end** of each work session
- **When changing implementation approach** or strategy

### 3. Implementation File Content Structure
```markdown
# Implementation Plan: [Jira Task Name] - [Brief Description]

## Overview
[Task description and objectives]

## Current Progress Summary
**Last Updated**: [Date]

### ✅ Completed Steps
[Numbered list of completed items]

### 🔄 In Progress
[Current work items]

### 📋 Pending Steps
[Future work items]

### 📊 Progress: X% Complete
[Progress breakdown by category]

## Requirements
[Detailed requirements from Jira task]

## Implementation Plan
[Detailed step-by-step implementation plan]

## Current Status & Best Practices
[Current implementation status and lessons learned]

## Files Modified
[List of all files changed]

## Next Steps
[Immediate next actions]

## Notes
[Additional context, decisions, or considerations]
```

### 4. Dual Update Requirement
- **ALWAYS update BOTH files** when making progress
- **memory.md**: For session-level progress and lessons learned
- **IMPLEMENTATION_[Task].md**: For task-specific detailed progress and implementation details

### 5. File Maintenance
- Keep implementation files **comprehensive and detailed**
- Include **code snippets** and **file references** where relevant
- Document **decisions made** and **rationale**
- Track **issues encountered** and **solutions applied**
- Maintain **chronological order** of progress

## Examples

### Good Implementation File Updates
- "Updated Phase 2.1 to reflect backend scanBoxId generation implementation"
- "Added console.log debugging statements to EstateService.createEstate() method"
- "Fixed frontend mutation to include scanBoxId in response fields"

### Good Memory Updates
- "Completed comprehensive field analysis for AE-1159"
- "Created patch file with all changes for backup"
- "Discovered scanBoxId generation already implemented in backend"

## Benefits
- **Task-specific tracking**: Detailed progress for each Jira task
- **Comprehensive documentation**: Both high-level and detailed progress
- **Knowledge preservation**: Implementation decisions and rationale
- **Future reference**: Easy to resume work on specific tasks
- **Team collaboration**: Clear progress visibility for other developers

## Enforcement
- This rule should be followed for **all Jira task work**
- Implementation files should be **created at the start** of task work
- Both files should be **updated together** during work sessions
- Files should be **kept current** and **comprehensive**
