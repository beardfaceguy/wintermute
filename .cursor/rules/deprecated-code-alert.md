# Deprecated Code Alert Rule

## Rule: ALWAYS ALERT WHEN WORKING WITH DEPRECATED CODE

### Core Principle
**When investigating task implementation, if you encounter deprecated code that needs modification, immediately bring it to the user's attention for team discussion or explicit approval.**

### Why This Matters
- **Deprecated Code**: Code marked as deprecated is typically scheduled for removal or replacement
- **Team Coordination**: Changes to deprecated code may conflict with ongoing migration efforts
- **Risk Management**: Modifying deprecated code could break future migration plans
- **Architectural Decisions**: Deprecated code changes require explicit team approval

### Implementation Guidelines

#### ✅ DO: Alert and Ask
- **Immediately stop** when encountering deprecated code that needs modification
- **Document the deprecated code** and explain why it needs to be changed
- **Ask for explicit approval** before proceeding
- **Request team discussion** if the change affects migration plans
- **Get developer explanation** of why the deprecated area is safe to work in

#### ❌ DON'T: Assume or Proceed
- Never assume deprecated code is safe to modify
- Never proceed with changes to deprecated code without approval
- Never ignore deprecation warnings or comments
- Never make "improvements" to deprecated code without explicit request

### Alert Process

When encountering deprecated code that needs modification:

1. **STOP implementation**
2. **Document the deprecated code**:
   - File path and line numbers
   - Deprecation warning/comment
   - Why the change is needed for the task
3. **Ask for explicit approval**:
   - Is this deprecated area safe to work in?
   - Should this be brought to the team for discussion?
   - Is there an alternative approach that avoids deprecated code?
4. **Wait for approval** before proceeding
5. **Document the decision** in implementation files

### Examples

#### ❌ Bad: Silent Modification of Deprecated Code
```typescript
// Task: Add scanBoxId to HeaderSelectedEstateId
// BAD: Silently changing deprecated getEstate2 to getEstate
const { data } = useGetEstateQuery({...}); // Ignores deprecation
```

#### ✅ Good: Alert About Deprecated Code
```typescript
// Task: Add scanBoxId to HeaderSelectedEstateId
// GOOD: Alert about deprecated getEstate2 usage
// ALERT: getEstate2 is marked as deprecated but is used by 37 pages
// Should we modify deprecated code or find alternative approach?
```

### Deprecation Indicators

Watch for these signs of deprecated code:
- **Comments**: `@deprecated`, `// DEPRECATED`, `// TODO: Remove`
- **Function names**: `oldFunction`, `legacyFunction`, `deprecatedFunction`
- **File names**: `legacy-`, `old-`, `deprecated-`
- **Import warnings**: "This import is deprecated"
- **TypeScript warnings**: "This type is deprecated"
- **Documentation**: Explicit deprecation notices

### Related Rules
- **Minimal Changes Principle**: Deprecated code changes often violate minimal changes
- **Requirement Clarification**: Never assume deprecated code is safe to modify
- **Implementation Tracking**: Document deprecated code decisions in implementation files

### Enforcement
- **Pre-implementation**: Scan for deprecated code in affected areas
- **During development**: Stop and alert when encountering deprecated code
- **Code review**: Flag any changes to deprecated code without approval
- **Documentation**: Always document deprecated code decisions


