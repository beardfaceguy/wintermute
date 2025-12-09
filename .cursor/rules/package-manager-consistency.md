# Package Manager Consistency Rule

## Rule: Always Use Yarn Package Manager

### Mandatory Requirement
**ALL repositories in this project MUST use yarn as the package manager. NEVER use npm.**

### Scope
This rule applies to:
- **alix-api** (backend repository)
- **alix-estate-manager** (frontend repository)
- **Any other repositories** in the project

### Commands to Use
**ALWAYS use yarn commands:**
- `yarn install` (not `npm install`)
- `yarn add package-name` (not `npm install package-name`)
- `yarn remove package-name` (not `npm uninstall package-name`)
- `yarn start`, `yarn dev`, `yarn build` (not `npm run start`, etc.)

### Commands to NEVER Use
**NEVER use npm commands:**
- `npm install`
- `npm install package-name`
- `npm uninstall package-name`
- `npm run start`, `npm run dev`, etc.

### Why This Rule Exists
1. **Team Standard**: The development team has standardized on yarn
2. **CI/CD Pipeline**: Build and deployment pipelines are configured for yarn
3. **Lock File Conflicts**: npm creates `package-lock.json`, yarn uses `yarn.lock`
4. **Dependency Resolution**: Different package managers resolve dependencies differently
5. **Deployment Failures**: Mixed package managers cause CI/CD failures and deployment issues

### Consequences of Violation
Using npm instead of yarn can cause:
- **Build failures** in CI/CD pipelines
- **Dependency resolution conflicts**
- **Deployment issues** in QA and production environments
- **Team workflow disruption**

### Exception Handling
**If npm is absolutely required for a specific operation:**
1. **Document the reason** in the implementation plan
2. **Coordinate with the team** before proceeding
3. **Ensure yarn.lock is not modified** by npm operations
4. **Clean up any npm artifacts** (package-lock.json) immediately

### Verification
**Before committing any changes:**
1. **Check that yarn.lock exists** and is up to date
2. **Verify no package-lock.json** files are present
3. **Confirm all dependency changes** were made with yarn commands
4. **Test locally** with yarn commands before pushing

### Implementation
**When working on this project:**
1. **Always start with yarn commands**
2. **If you accidentally use npm**, immediately revert and use yarn
3. **Update dependencies** only with yarn commands
4. **Document any package manager issues** in the implementation plan

---

**This rule is MANDATORY and must be followed by all Cursor Agents working on this project.**
