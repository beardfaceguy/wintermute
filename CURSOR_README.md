
# Cursor Agent Guidelines

## For Cursor Agents

This project uses a comprehensive `.cursor/` directory system to maintain consistent AI assistant behavior and project context. **All Cursor Agents working on this project MUST follow the rules and guidelines defined in the `.cursor/` directory.**

## .cursor Directory Overview

The `.cursor/` directory serves as a comprehensive knowledge base and rule system for AI-assisted development on the Alix Estate Manager project. It contains:

### 📁 **rules/** - AI Assistant Behavior Rules
- **11 specialized rule files** governing different aspects of AI behavior
- **MANDATORY COMPLIANCE** - these rules MUST be followed and kept in context
- Covers command execution, memory management, package manager consistency, code quality validation, minimal changes principle, deprecated code alerts, AI response boundaries, and safety protocols
- **CRITICAL**: Always keep `.cursor/rules/` directory contents in context for every session

### 📁 **memory/** - Cross-Cutting Insights  
- **`memory.md`** - Comprehensive cross-cutting insights and reusable knowledge (422 lines)
- **Updated when discovering** architecture patterns, environment gotchas, or lessons learned
- Maintains valuable knowledge across multiple implementations and sessions
- Contains detailed environment configurations, authentication info, and critical commands

### 📁 **docs/** - Project Documentation
- **Implementation plans** - Detailed implementation plans for specific tasks organized by Jira ticket (e.g., AE-1222/)
- **Setup guides** - Environment setup and dependency resolution guides
- **Discovery Agent Analysis** - Comprehensive analysis of AI-powered document processing system
- **CodeRabbit Rules Reference** - Detailed coding standards and compliance rules
- **ALIX Coding Practices** - Best practices derived from PR reviews
- **research_repos/** - Reference snapshots of related repositories (alix-api, alix-estate-manager) for research and context

### 📁 **patches/** - Backup and Recovery
- Currently empty directory for storing code patches and rollback capabilities

## Critical Rules for Cursor Agents

### 🚨 **MANDATORY RULES**
1. **Memory Tracking**: Update `.cursor/memory/memory.md` with cross-cutting insights and reusable knowledge
2. **Implementation Tracking**: Update both memory.md and implementation files together
3. **Command Safety**: Never use unescaped `!` in shell commands
4. **Interactive Commands**: Use non-interactive flags (`--no-pager`, `| cat`, etc.)
5. **Package Manager Consistency**: ALWAYS use yarn, NEVER use npm
6. **Pre-Commit Validation**: ALWAYS run `yarn typecheck` and `yarn lint` before every commit
7. **Requirement Clarification**: NEVER assume requirements or add "common practices" without explicit user approval
8. **Minimal Changes Principle**: ALWAYS GO FOR THE MINIMUM CHANGES NEEDED TO COMPLETE THE TASK - Never make architectural changes that affect many components without explicit approval
9. **Deprecated Code Alert**: ALWAYS ALERT WHEN WORKING WITH DEPRECATED CODE - Stop and ask for team discussion or explicit approval before modifying deprecated code
10. **AI Response Boundaries**: Clearly distinguish between capabilities and actions in AI responses
11. **Git Review Tagging**: Use appropriate git tags for review milestone tracking

### 📋 **Project Status**
- Check `.cursor/memory/memory.md` for cross-cutting insights and patterns
- Review `.cursor/docs/` for specific implementation plans
- Follow established patterns documented in the implementation files

## Quick Start for New Agents

1. **🚨 CRITICAL**: Always keep `.cursor/rules/` directory contents in context for every session
2. **Read the rules**: Start with `.cursor/rules/README.md` for complete rule overview
3. **Check memory**: Review `.cursor/memory/memory.md` for cross-cutting insights and patterns
4. **Review implementation**: Read relevant files in `.cursor/docs/` for detailed plans
5. **Follow patterns**: Use established implementation patterns from documentation
6. **Update memory**: Add cross-cutting insights to `.cursor/memory/memory.md` when discovered

## Project Structure

```
[Project Root]/
├── README.md                               # Generic Cursor Agent guidelines
└── .cursor/                                # AI Assistant Rules & Documentation
    ├── docs/                               # Documentation directory
    │   ├── AE-1222/                       # Task-specific documentation directory
    │   ├── ALIX_coding_practices.md       # Best practices from PR reviews
    │   ├── CODE_RABBIT_RULES.md           # CodeRabbit compliance rules
    │   ├── DISCOVERY_AGENT_ANALYSIS_LOG.md # Discovery Agent analysis work log
    │   ├── Discovery_Agent_summary.md      # Discovery Agent comprehensive summary
    │   ├── DISCOVERY_AGENT_SETUP.md       # Discovery Agent setup guide
    │   └── research_repos/                 # Reference repositories
    │       ├── alix-api/                   # Main API repository snapshot
    │       ├── alix-estate-manager/        # Frontend repository snapshot
    │       └── box-typescript-sdk-gen/    # Box SDK repository snapshot
    ├── memory/                            # Cross-cutting insights
    │   └── memory.md                      # Cross-cutting insights and reusable knowledge (422 lines)
    ├── patches/                           # Backup and patch files (currently empty)
    └── rules/                             # AI assistant behavior rules (11 files)
        ├── README.md                      # Rules overview and usage guidelines
        ├── capability-vs-action-distinction.md # Rule for AI response boundaries
        ├── deprecated-code-alert.md       # Rule for alerting about deprecated code changes
        ├── git-tagging-for-review.md      # Rule for git review milestone tracking
        ├── implementation-tracking.md     # Rule for Jira task progress tracking
        ├── interactive-commands.md        # Rule for non-interactive command execution
        ├── memory-tracking.md             # Rule for maintaining session memory
        ├── minimal-changes-principle.md  # Rule for minimal scope changes
        ├── no-unescaped-exclamation.md   # Rule for safe text output
        ├── package-manager-consistency.md # Rule for yarn-only package management
        ├── pre-commit-validation.md       # Rule for code quality validation
        └── requirement-clarification.md  # Rule for requirement clarification
```

## Environment Requirements

- Check `.cursor/docs/DISCOVERY_AGENT_SETUP.md` for Discovery Agent environment setup
- Review `.cursor/memory/memory.md` for environment gotchas and solutions
- Follow setup guides in `.cursor/docs/` for proper configuration
- Reference `.cursor/docs/research_repos/` for understanding related repository dependencies

---

**⚠️ IMPORTANT**: This project has specific rules and patterns that have been developed through extensive analysis. Always follow the `.cursor/` directory guidelines to maintain consistency and avoid repeating previous mistakes.

**📚 For detailed information**: See `.cursor/rules/README.md` for complete rule explanations and `.cursor/docs/` for specific implementation plans and setup guides.