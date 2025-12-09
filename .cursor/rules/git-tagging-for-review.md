# Git Tagging for Code Review Tracking

## Purpose
This rule defines the standard method for tagging Git revisions to mark them as "ready for code review" or "passed local tests" for personal tracking and future reference.

## Tagging Method

### When to Use
- When a revision has passed local tests
- When code is ready for code review
- When a deployment has been successfully tested
- For marking important milestones in development

### Tagging Command
```bash
# Create lightweight tag with date
git tag AE-[TASK-NUMBER]-ready-for-review-$(date +%Y%m%d)

# Push tag to remote
git push origin AE-[TASK-NUMBER]-ready-for-review-$(date +%Y%m%d)
```

### Tag Naming Convention
- Format: `AE-[TASK-NUMBER]-[STATUS]-[YYYYMMDD]`
- Examples:
  - `AE-1159-ready-for-review-20250913`
  - `AE-1159-qa3-deployed-20250913`
  - `AE-1159-tested-20250913`

### Usage Examples
```bash
# List all tags for a specific task
git tag -l AE-1159*

# Show details of a tagged revision
git show AE-1159-ready-for-review-20250913

# Checkout a tagged revision
git checkout AE-1159-ready-for-review-20250913

# Compare tagged revisions
git diff AE-1159-ready-for-review-20250913 AE-1159-tested-20250913
```

## Implementation Notes
- Use lightweight tags (not annotated) for simplicity
- Always push tags to remote for backup
- Include date in tag name for chronological tracking
- Use consistent naming convention across all repositories

## Benefits
- Easy identification of tested/ready revisions
- Quick rollback to known good states
- Historical tracking of development milestones
- Personal reference without cluttering commit history
