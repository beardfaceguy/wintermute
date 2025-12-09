# Interactive Commands Rule

## Rule: Use Non-Interactive Flags for Commands

**MANDATORY**: Check informational commands like 'systemctl status' to see if they run in interactive mode by default and find their flag to run in non-interactive mode.

### When This Rule Applies:
- **System commands** (`systemctl`, `journalctl`, `top`, `htop`, etc.)
- **Package managers** (`apt`, `yum`, `brew`, etc.)
- **Git commands** that might use pager (`git log`, `git diff`, etc.)
- **Any command** that might open an interactive interface

### Required Actions:
1. **Check command behavior** before running
2. **Find non-interactive flags** (e.g., `--no-pager`, `--non-interactive`, `-q`)
3. **Keep a list** of commands and their non-interactive flags in `/mnt/seagate/cursor/interactive_commands.md`
4. **Search the list first** before researching new commands
5. **Always use non-interactive flags** when available

### Special Cases:
- **systemctl status**: ALWAYS use `--no-pager` flag
- **git commands**: Use `| cat` to prevent pager
- **Package managers**: Use `-y` or `--yes` for auto-confirmation

### Enforcement:
This rule prevents commands from hanging in interactive mode and blocking communication.

---

**This rule ensures commands complete without user interaction.**


