# Minimal Changes Principle

## Rule: ALWAYS GO FOR THE MINIMUM CHANGES NEEDED TO COMPLETE THE TASK

### Core Principle
**Never make changes that affect more components than absolutely necessary to complete the assigned task.**

### Why This Matters
- **Testing Scope**: Changes that affect multiple components require retesting of all affected areas
- **Risk Management**: Larger changes introduce more potential for bugs and regressions
- **Team Coordination**: Major architectural changes require team discussion and approval
- **Scope Control**: Prevents feature creep and maintains focus on the specific task

### Implementation Guidelines

#### ✅ DO: Minimal Approach
- Add new fields to existing queries/components when possible
- Use existing patterns and structures
- Make changes only to components directly related to the task
- Document the minimal scope in implementation plans

#### ❌ DON'T: Architectural Changes
- Change core queries/components used by many other parts of the application
- Migrate to "better" patterns unless explicitly requested
- Make "improvements" that go beyond task requirements
- Assume that "newer" or "better" approaches are preferred

### Approval Process for Major Changes

If a task **cannot be completed** without making changes that affect many other components:

1. **Stop implementation**
2. **Document all affected areas** that would need testing
3. **Propose a two-step approach**:
   - Step 1: Separate task to change the component/query
   - Step 2: Add the new feature in a separate task
4. **Get explicit approval** before proceeding
5. **Update implementation plan** with approved approach

### Examples

#### ❌ Bad: Major Architectural Change
```typescript
// Task: Add scanBoxId to HeaderSelectedEstateId
// BAD: Changing from getEstate2 to getEstate affects 37 pages
const { data } = useGetEstateQuery({...}); // Affects entire app
```

#### ✅ Good: Minimal Change
```typescript
// Task: Add scanBoxId to HeaderSelectedEstateId  
// GOOD: Add scanBoxId to existing getEstate2 query
const { data } = useGetEstate2Query({...}); // Only affects this component
```

### Enforcement
- **Pre-implementation**: Review implementation plan for minimal approach
- **During development**: Question any changes that affect more than necessary
- **Code review**: Flag architectural changes that weren't explicitly requested
- **Documentation**: Always document the scope of changes made

### Related Rules
- **Requirement Clarification**: Never assume "improvements" are wanted
- **Implementation Tracking**: Document the minimal scope in implementation plans
- **Memory Tracking**: Record lessons learned about scope management


