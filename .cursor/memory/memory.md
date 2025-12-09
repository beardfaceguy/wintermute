# Memory - Alix Estate Manager Development Insights

## Architecture Patterns Discovered

### Field Implementation Patterns
- **Estate Email Pattern**: Field added directly to Estate model with @unique constraint, available for both creation and updates
- **taxId Pattern**: Field added via migration, nullable initially, unique constraint added later
- **scanBoxId Pattern**: Auto-generated on backend during estate creation, display-only in frontend
- **Deceased Fields Pattern**: Added to Deceased table via migration, handled in writeCoreEstateInformation mutation

### GraphQL Resolver Priority
- **Generated resolvers** are registered FIRST in resolver array
- **Custom resolvers** are registered LAST
- **Frontend mutations** may use generated resolvers instead of custom ones
- **Solution**: Add logic to BOTH generated and custom resolvers

### UI Conditional Rendering Patterns
- **Estate Email**: Always display with fallback "-" when empty
- **Test Account**: Boolean field with conditional styling
- **Status Fields**: Color-coded based on status values
- **Boolean Fields**: Checkbox or toggle display
- **Complex Conditional Logic**: Multiple conditions for field visibility
- **Fallback Display**: Show "-" or placeholder when field is null/empty
- **scanBoxId Pattern**: Conditional rendering - only display when value exists (no fallback)
- **Copy Functionality Pattern**: Use `navigator.clipboard.writeText()` with `useNotify` for user feedback

### Database Migration Strategy
- **Use postgres superuser** for schema changes when regular user lacks privileges
- **Follow established patterns** (taxId, Estate Email) for new field implementations
- **Apply pending migrations** before making schema changes
- **Create database backups** after migrations for quick restart capability

### Node Version Requirements
- **Critical**: Project requires Node 23+ to match deployment environment
- **Issue**: Node 20.18.1 causes generator failures and prevents client generation
- **Solution**: Default Node version set to 23 via `nvm alias default 23`
- **Status**: ✅ UPDATED - Default Node version now 23.11.1, matches QA3 deployment environment
- **Policy**: Local environment must match deployment servers (Node 23), not the other way around

### Prisma Version Requirements
- **Critical**: Project requires Prisma version `^5.18.0` (as specified in package.json)
- **Issue**: `typegraphql-prisma` only works with specific Prisma versions
- **Check**: Run `npm list prisma` to verify installed version
- **Fix**: If version mismatch, reinstall correct Prisma version with `npm install prisma@^5.18.0`

## Environment Gotchas

### Backend Startup Procedure
- **ALWAYS switch to Node 23 first**: `export NVM_DIR="$HOME/.config/nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && nvm use 23`
- **Then run in background**: `yarn start` (NEVER foreground - user loses control)
- **CRITICAL**: Wait 1-2 minutes for full startup - server does Prisma generation, service initialization, and database connections
- **Success indicator**: Look for "🚀 Server ready at http://localhost:8080/graphql" message
- **Frontend commands**: Use `npm run dev` (NOT `yarn start` - that command doesn't exist in frontend)

### Dependency Resolution
- **Backend (alix-api)**: **ALWAYS use yarn** - never use npm
- **Frontend (alix-estate-manager)**: **ALWAYS use yarn** - never use npm  
- **Package manager consistency**: Mixed npm/yarn causes CI/CD failures and deployment issues
- **MUI Lab conflicts**: React 19 vs MUI Lab compatibility resolved with legacy peer deps
- **Backend requires Node 23+**: Use nvm to switch versions

### Service Management
- **NEVER run services in foreground**: Always use `is_background: true` or user will need to kill them manually
- **Multiple instances**: Old ts-node-dev processes weren't properly killed, causing conflicts
- **Clean startup**: Kill all existing processes before restarting services

### Database User Setup
- **patrickclawson user**: Requires specific permissions for Prisma migrations
- **Permission setup commands** (run immediately after database restore):
  ```sql
  -- Grant CREATEDB privilege for shadow database
  ALTER USER patrickclawson WITH CREATEDB;
  
  -- Grant ownership of all enum types
  DO $$ DECLARE r RECORD; 
  BEGIN 
    FOR r IN SELECT typname FROM pg_type WHERE typtype = 'e' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public') 
    LOOP 
      EXECUTE 'ALTER TYPE "' || r.typname || '" OWNER TO patrickclawson;'; 
    END LOOP; 
  END $$;
  
  -- Grant ownership of all tables
  DO $$ DECLARE r RECORD; 
  BEGIN 
    FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' 
    LOOP 
      EXECUTE 'ALTER TABLE "' || r.tablename || '" OWNER TO patrickclawson;'; 
    END LOOP; 
  END $$;
  
  -- Grant all privileges
  GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO patrickclawson;
  GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO patrickclawson;
  GRANT ALL PRIVILEGES ON TABLE _prisma_migrations TO patrickclawson;
  ```

## Key Lessons Learned

### Implementation Methodology
- **Jira-First Analysis**: MUST check Jira task requirements first before analyzing UI patterns
- **Complete Field Analysis**: MUST analyze ALL existing fields before implementing new ones to understand patterns
- **Backend-First Verification**: Always check if functionality already exists in backend services
- **Complete Data Flow Tracing**: Database → Backend Service → GraphQL → Frontend mutation → UI display
- **CRITICAL: Requirement Clarification**: NEVER assume requirements or add "common practices" without explicit user approval
- **MAJOR LESSON: Unauthorized Character Exclusions**: Added character exclusions (I,O,0,1) to scanBoxId without user approval - major overstep that violated requirement clarification rule
- **CRITICAL: Minimal Changes Principle**: ALWAYS GO FOR THE MINIMUM CHANGES NEEDED TO COMPLETE THE TASK - Never make architectural changes that affect many components without explicit approval
- **RECENT EXAMPLE**: After successfully implementing backOff for scanBoxId retry logic, agent started investigating other retry patterns in the file - this was scope creep beyond the specific task requirement
- **CRITICAL: Deprecated Code Alert**: ALWAYS ALERT WHEN WORKING WITH DEPRECATED CODE - Stop and ask for team discussion or explicit approval before modifying deprecated code

## ALIX Coding Strategies (From PR Reviews)

### Database-First Philosophy
- **Prefer database-native solutions** over application-level complexity
- **Use PostgreSQL sequences** for auto-incrementing values when appropriate
- **Leverage database functions** like `lpad()` and `to_hex()` for formatting
- **Use `DEFAULT` values** to handle automatic generation at the database level
- **Prefer declarative solutions** over imperative application code
- **Consider the database as part of the solution** rather than just storage

### Simplicity Over Complexity
- **Keep implementations simple** - if the database can handle it, let it
- **Avoid over-engineering** when database features can solve the problem elegantly
- **Prefer simpler implementations** over complex custom solutions
- **Avoid creating services for simple operations** that can be handled inline
- **Consider if a service adds value** or just adds unnecessary complexity
- **Use built-in JavaScript methods** instead of manual character iteration
- **Avoid unnecessary complexity** - if a one-liner can achieve the same result, prefer it

### Prisma ORM Standards
- **Use Prisma ORM methods** instead of raw SQL for simple database operations
- **Prefer `tx.modelName.find()`** over `tx.$queryRaw` when the operation is straightforward
- **Use Prisma's fluent API** instead of raw SQL for database operations
- **Leverage Prisma's query optimization** capabilities
- **Use Prisma's type-safe query builders** for better maintainability
- **Reserve raw SQL** only for complex operations that cannot be expressed with Prisma ORM
- **Maintain consistency** with the codebase's ORM usage patterns

### Database Schema Standards
- **All models primary/foreign keys must be in UUID format** (`id String @id @default(uuid())`)
- **Use appropriate database data types** to enforce constraints rather than relying on CHECK constraints
- **Avoid creating separate counter tables** when the unique constraint on the target field already provides the necessary uniqueness guarantee
- **Follow established naming conventions** for database tables (not snake_case)
- **Use database transactions** for operations that must be atomic
- **Include ID generation in the same transaction** as the main operation

### Code Quality and Maintenance
- **Remove unused files** when implementation changes make them obsolete
- **Keep codebase clean** by eliminating dead code
- **Avoid obvious comments** that simply restate what the code is doing
- **Comments should add value** by explaining "why" not "what"
- **Remove unnecessary error handlers** that handle edge cases that shouldn't occur in production
- **Use database seeding** instead of runtime initialization methods for initial data
- **Don't use error handling as migration reminders** - handle migrations properly during deployment

### Service Architecture Patterns
- **Use instance-based services** instead of static methods when following service pattern
- **Initialize dependencies in constructor** for better organization and maintainability
- **Follow dependency injection pattern** for better testability and modularity
- **Reserve services for complex business logic** that requires multiple steps or external dependencies
- **Consider multiple approaches** before settling on a complex solution
- **Evaluate external libraries** like CUID2 for ID generation when appropriate

### Error Handling and Business Logic
- **Consider business impact** of error conditions and limits
- **Plan for edge cases** that could block core business functionality
- **Ensure graceful degradation** or alternative approaches when limits are reached
- **Database schema issues** should be handled through proper migrations during deployment, not through runtime error handling
- **The application should assume the database schema is correct and up-to-date**

### Code Analysis Requirements
- **MUST examine git history** for each field type to see actual implementation approach
- **MUST check backend services** not just GraphQL schema - business logic lives in services
- **MUST verify frontend mutations** request all needed fields in response
- **MUST understand conditional rendering patterns** for UI behavior decisions

### Common Issues and Solutions
- **"Crud resolvers" confusion**: Generated resolvers take precedence over custom ones
- **Frontend mutation missing fields**: Check if mutation response includes all needed fields
- **Database migration drift**: Use proper Prisma migrations, not db push
- **Incomplete rollback**: Verify ALL files are clean after rollback, not just git status
- **Scope Creep Prevention**: Always ask "What's the minimal change needed?" before implementing architectural changes
- **Deprecated Code Awareness**: Always scan for and alert about deprecated code that needs modification

## Critical Commands

### Backend Startup
```bash
# Switch to Node 22
export NVM_DIR="$HOME/.config/nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && nvm use 22

# Start backend in background
yarn start
```

### Frontend Startup
```bash
# Install dependencies
npm install --legacy-peer-deps

# Start development server
npm run dev
```

### Database Operations
```bash
# Check migration status
npx prisma migrate status

# Apply pending migrations
npx prisma migrate deploy

# Regenerate Prisma client
npx prisma generate

# Create database backup
pg_dump -h localhost -p 5432 -U patrickclawson -d alix > alix_backup.sql
```

### Service Management
```bash
# Kill all backend processes
pkill -f "ts-node-dev"
pkill -f "yarn start"

# Check running services
ps aux | grep -E "(yarn|ts-node|npm)"
```

## Authentication Information

### Working Credentials
- **Email**: `admintest@meetalix.com`
- **Password**: `te8mAlix!`
- **Role**: Admin
- **Site URL**: http://localhost:3000/

### Other Available Users
- `david+admin@meetalix.com` (SuperAdmin) - Password: Unknown
- `david+testofferson@meetalix.com` (Admin) - Password: Unknown
- `vyacheslav.solomin+admin@meetalix.com` (SuperAdmin) - Password: Unknown
- `travis+admin@meetalix.com` (SuperAdmin) - Password: Unknown

### Access Requirements
- **Authentication Required**: Yes - AWS Cognito authentication required
- **Public Pages**: Only login, forgot-password, and set-new-password pages
- **Required Roles**: Users must have 'Admin' or 'SuperAdmin' role

## Environment Configuration

### Frontend Environment Variables
```bash
# AWS Cognito Authentication (REQUIRED)
VITE_COGNITO_USERPOOL_ID=your_user_pool_id
VITE_COGNITO_CLIENT_ID=your_client_id  
VITE_IDENTITY_POOL_ID=your_identity_pool_id

# GraphQL Backend (REQUIRED)
VITE_GRAPHQL_ENDPOINT=http://localhost:8080/graphql
VITE_API_ENDPOINT=http://localhost:8080

# Development Settings
VITE_APP_VERSION=local
VITE_AWS_REGION=us-east-1
VITE_OPTIMIZELY_SDK_KEY=your_optimizely_key
```

### Backend Environment Variables
```bash
# Database Connection (REQUIRED)
DATABASE_URL=postgres://patrickclawson:postgres@127.0.0.1:5432/alix

# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# External Services (placeholders acceptable for development)
CLICKUP_API_KEY=your_clickup_key
PLAID_CLIENT_ID=your_plaid_client_id
PLAID_SECRET=your_plaid_secret
BOX_CLIENT_ID=your_box_client_id
BOX_CLIENT_SECRET=your_box_client_secret
```

## Documentation Preferences

### Markdown Formatting
- **No emojis**: Do not use emojis in .md files
- **Clean formatting**: Use standard markdown syntax only
- **Professional appearance**: Maintain clean, readable documentation

## Current Environment Status

### Services Status
- **Frontend**: Running on port 3000 (React application)
- **Backend**: Running on port 8080 (GraphQL endpoint)
- **Database**: PostgreSQL running on port 5432 with staging data (117 estates)
- **Node**: Version 22 active via nvm
- **Yarn**: Installed globally and working

### Database Status
- **PostgreSQL**: Container running on port 5432
- **Database**: `alix` accessible with 117 estates
- **User**: `patrickclawson` with proper permissions
- **Migrations**: All 30 pending migrations applied successfully
- **Backup**: Available for quick restart capability

### Prisma Status
- **Version**: 5.22.0 (downgraded from 6.15.0 for typegraphql-prisma compatibility)
- **GraphQL Schema**: ✅ Successfully regenerated with scanBoxId field
- **Estate Model**: ✅ Includes `scanBoxId?: string | null;`
- **EstateCreateInput**: ✅ Includes `scanBoxId?: string | undefined;`
- **Status**: Phase 2.3 completed successfully

### Backend Testing Results ✅ COMPLETED
- **Test Estates Created**: 2 (Henry Testerton, Jane Testerson)
- **scanBoxId Generation**: ✅ Working correctly
- **Format Validation**: ✅ 8-digit alphanumeric (A-Z, 0-9) - CORRECTED to match original requirements
- **Uniqueness**: ✅ Different IDs generated (LWXUUKFU, MQQ5E9PG)
- **Database Storage**: ✅ Confirmed in PostgreSQL
- **GraphQL Integration**: ✅ scanBoxId included in responses
- **Authentication**: ✅ Master token working for testing
- **CodeRabbit Compliance**: ✅ All fixes implemented and reviewed

## Discovery Agent Analysis Insights (2025-01-16)

### Discovery Agent Architecture Patterns
- **Multi-Stage Pipeline**: PDF → Images → Classification → Fact Extraction → Database Storage
- **LangGraph Workflows**: Complex AI processing orchestration with conditional routing
- **Agent-Based Processing**: Specialized AI agents for different tasks (classifier, extractor, grouper)
- **Chunked Processing**: Handles large document sets efficiently with configurable chunk sizes
- **Distributed Locking**: Prevents concurrent modifications using DynamoDB-based locks

### AI Agent Patterns
- **Specialized Agents**: Each agent has specific purpose (page classification, fact identification, data extraction)
- **Prompt Engineering**: System instructions tailored for estate settlement expertise
- **Confidence Scoring**: All extractions include confidence levels for quality assessment
- **Context-Aware Processing**: Agents receive deceased name and document context
- **Model Selection**: Different models for different tasks (GPT-4.1-mini for classification, GPT-4o-mini for reasoning)

### Document Processing Patterns
- **File Type Handling**: Supports PDF (converted to images) and direct image files
- **Page Splitting**: Multi-page documents split into individual pages for processing
- **Document Grouping**: Pages grouped back into complete documents using AI reasoning
- **Metadata Extraction**: Institution, date, document type extracted for all documents
- **Supporting Document Linking**: Processed files linked to extracted entities

### Data Extraction Patterns
- **Entity-Based Extraction**: Assets, debts, contacts, obligations extracted separately
- **Relationship Mapping**: Entities linked to supporting documents and related entities
- **Duplicate Detection**: Primary key matching (needs improvement to vector-based reasoning)
- **Event Tracking**: Data enrichment events created for audit trails
- **Graceful Degradation**: Continues processing even if individual files fail

### Current Implementation Gaps
- **Document Organization**: Missing standardized naming convention "{Institution-NameDocument-Type-MMDD-YYYY}"
- **Folder Structure**: No organized subfolder system for different document types
- **Human Review**: No CoPilot system for care team oversight
- **Quality Assurance**: No automated testing against known good results
- **DoD Value Handling**: Doesn't distinguish between date of death and current values
- **Enhanced Reasoning**: Needs better "does this represent new estate information" logic

### Business Logic Requirements
- **Priority-Based Processing**: P1 (common, multiple instances), P2 (common, single instance), P3 (less common)
- **Context Awareness**: Distinguish between decedent and beneficiary information
- **Expense Classification**: Separate reimbursable from non-reimbursable expenses
- **External Integration**: Planned integration with unclaimed property databases, credit reports, etc.

### Technical Dependencies
- **AWS Services**: S3 (storage), DynamoDB (locking) - Textract confirmed unused
- **External APIs**: Box (file storage), OpenAI (AI models), Langfuse (tracking)
- **Database**: PostgreSQL with GraphQL layer for data persistence
- **Processing**: Poppler for PDF conversion, Sharp for image processing

### Performance Considerations
- **Parallel Processing**: Multiple files processed simultaneously
- **Memory Management**: Efficient image conversion and temporary file cleanup
- **Error Handling**: Comprehensive error tracking and recovery
- **Monitoring**: Langfuse integration for AI operation tracking

### Integration Opportunities
- **Plaid Integration**: Bank account access (estimated 50-70% of population)
- **State Databases**: MissingMoney.com for unclaimed property
- **Federal Sources**: SSA earnings reports, IRS transcripts, credit reports
- **Insurance Locators**: NAIC Life Policy Locator
- **Data Aggregators**: TLOx, idiCORE, impact360

## Discovery Agent Testing & Setup Status (2025-01-16)

### Environment Configuration ✅ COMPLETED
- **Environment Variables**: All required variables configured in `environment_test` file
- **API Integration**: Successfully authenticated with alix-api using GraphQL
- **Estate Creation**: Created test estate for Patrick Clawson (ID: `4ac3787d-3fa0-46f7-af63-c4782d6732d2`)
- **Database Status**: All 242 migrations applied successfully
- **Discovery Agent Tables**: Present and ready (DiscoveryAgentRunHistory, DiscoveryAgentFileRunInfo, RemoteFileInfo)

### AI Agent Testing ✅ COMPLETED
- **Page Classifier**: Successfully tested with real PDF documents
- **Fact Identifier**: Successfully tested with extracted text content
- **GPT-4 Vision**: Confirmed working for image analysis and OCR
- **GPT-4 Text**: Confirmed working for structured fact extraction
- **Classification Accuracy**: Verified against actual document descriptions

### Test Documents Available
- **scanned_docs1.pdf**: Multiple cards (driver's license, SSN, immunization records) - intentionally tricky
- **scanned_doc2.pdf**: Property tax statement for Corinne and Patrick Clawson
- **scanned_doc3.pdf**: Marriage certificate for Patrick and Corinne Clawson
- **scanned_doc4.pdf**: California High School Proficiency Exam - "junk" document
- **scanned_doc5.pdf**: Clawson Family Trust pages 1 and 2

### Ready for Full Pipeline Testing
- **Individual Agents**: ✅ Tested and working
- **Database Schema**: ✅ Up to date with all Discovery Agent tables
- **Estate Created**: ✅ Patrick Clawson estate ready for document processing
- **Environment**: ✅ All services configured and accessible
- **Next Step**: Full Discovery Agent pipeline test with real documents

---

*Last updated: 2025-01-16*
*This file contains cross-cutting insights useful across multiple implementations and sessions.*