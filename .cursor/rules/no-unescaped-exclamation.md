# No Unescaped Exclamation Marks Rule

## Rule: Don't Use Unescaped Exclamation Marks in Print Messages

**MANDATORY**: Don't use unescaped exclamation marks ("!") in print messages.

### When This Rule Applies:
- **All print statements** and console output
- **Error messages** and logging
- **User-facing text** in applications
- **Any string output** that might be processed by shells or scripts

### Required Actions:
1. **Escape exclamation marks** when necessary: `"Hello world\\!"`
2. **Use alternative punctuation** when possible: `"Hello world."`
3. **Avoid exclamation marks** in shell commands and scripts
4. **Test output** to ensure it doesn't cause shell interpretation issues

### Examples:
- ✅ `console.log("Operation completed successfully.")`
- ✅ `console.log("Warning: Check your input!")` (if properly escaped)
- ❌ `console.log("Operation completed!")` (unescaped in shell context)

### Enforcement:
This rule prevents shell interpretation issues and ensures clean, safe output in all contexts.

---

**This rule ensures safe and clean text output without shell interpretation conflicts.**


