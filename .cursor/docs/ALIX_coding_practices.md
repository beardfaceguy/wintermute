# ALIX Coding Practices

This document captures coding practices and standards derived from PR reviews and team discussions.

## Contents
- [PR Review Comments](#pr-review-comments)
- [Best Practices](#best-practices)
- [Architecture Guidelines](#architecture-guidelines)

## PR Review Comments

### Database Schema Standards

**PR Review Comment from @s1owjke:**
> All our models primary/foreign keys must be in uuid format
> 
> ```prisma
> id String @id @default(uuid())
> ```

**Context:** This comment was made regarding the `ScanBoxIdCounter` model in `prisma/schema.prisma` where the primary key was defined as `Int` instead of `String` with UUID format.

**Practice:** All database models in ALIX must use UUID format for primary keys, not integer IDs.

### Database Data Types and Constraints

**PR Review Comment from @s1owjke:**
> Here, you should use the correct data type, don't rely on CHECK if it could be handled with data type

**Context:** This comment was made regarding the `scan_box_id_counter` migration where a CHECK constraint was used to enforce the 8-hex limit (0..4294967295) instead of using an appropriate data type.

**Practice:** Use appropriate database data types to enforce constraints rather than relying on CHECK constraints when the data type itself can handle the validation.

**PR Review Comment from @s1owjke:**
> This one must be uuid, not integer

**Context:** This comment was made regarding the `scan_box_id_counter` table creation where the `id` field was defined as `INTEGER` instead of UUID format.

**Practice:** All primary key fields must use UUID format, not integer types, consistent with ALIX's database schema standards.

**PR Review Comment from @s1owjke:**
> It doesn't make any sense to have detached record (remember that scanBoxId is unique in your schema)

**Context:** This comment was made regarding the initial counter value insertion in the migration, questioning the need for a separate counter table when `scanBoxId` already has a unique constraint in the schema.

**Practice:** Avoid creating separate counter tables when the unique constraint on the target field already provides the necessary uniqueness guarantee. Consider whether detached records are necessary or if the existing schema constraints are sufficient.

### Error Handling and Deployment

**PR Review Comment from @s1owjke:**
> We shall not have these checks, it must be handled with migrations during deployment

**Context:** This comment was made regarding error handling in `ScanBoxIdService.ts` where code was checking for "table does not exist" errors and handling them gracefully.

**Practice:** Database schema issues (like missing tables) should be handled through proper migrations during deployment, not through runtime error handling in application code. The application should assume the database schema is correct and up-to-date.

### Code Simplicity and Optimization

**PR Review Comment from @s1owjke:**
> Suboptimal, please implement it other way, without iterator

**Context:** This comment was made regarding a `generateScanBoxId()` function that used a loop with string concatenation to build an 8-character ID, suggesting a simpler one-liner approach using `Math.random().toString(36).substring(2, 10)`.

**Practice:** 
- **Prefer simpler implementations** over complex loops when possible
- **Use built-in JavaScript methods** (`toString(36)`) instead of manual character iteration
- **Avoid unnecessary complexity** - if a one-liner can achieve the same result, prefer it
- **Consider readability vs. performance** - simpler code is often more maintainable

### Database Naming Conventions

**PR Review Comment from @Selnapenek:**
> we are not using snake_case for tables names

**Context:** This comment was made regarding the `@@map("scan_box_id_counter")` directive in the Prisma schema, where the table name was using snake_case formatting.

**Practice:** ALIX does not use snake_case for table names. Follow the established naming convention for database tables (likely camelCase or PascalCase based on the existing schema patterns).

### Code Comments and Documentation

**PR Review Comment from @Selnapenek:**
> unnecessary comments

**Context:** This comment was made regarding comments in `EstateService.ts` that explained obvious functionality like "scanBoxId is auto-generated, no validation needed" and "Generate scanBoxId using atomic counter (guaranteed unique)".

**Practice:** 
- **Avoid obvious comments** that simply restate what the code is doing
- **Comments should add value** by explaining "why" not "what"
- **Remove redundant comments** that don't provide additional context or reasoning
- **Let the code be self-documenting** through clear variable names and function names

### Service Pattern and Class Instantiation

**PR Review Comment from @Selnapenek:**
> if you go with service pattern and classes please create and use instance. creating and using instances is crucial for better organization and maintainability in our codebase.

**Context:** This comment was made regarding the use of static methods in `ScanBoxIdService` instead of creating instances. The reviewer suggested using dependency injection pattern with constructor initialization.

**Practice:** 
- **Use instance-based services** instead of static methods when following service pattern
- **Initialize dependencies in constructor** for better organization and maintainability
- **Follow dependency injection pattern** for better testability and modularity
- **Create service instances** rather than calling static methods directly

### Error Handling and Dead Code

**PR Review Comment from @Selnapenek:**
> unnessesary handler please remove it. this is a deadcode please don't use this pattern as migration todo dev notification.

**Context:** This comment was made regarding error handling code in `ScanBoxIdService.ts` that checked for "table does not exist" errors and logged migration-related messages.

**Practice:** 
- **Remove unnecessary error handlers** that handle edge cases that shouldn't occur in production
- **Avoid dead code patterns** that serve as development notifications
- **Don't use error handling as migration reminders** - handle migrations properly during deployment
- **Keep error handling focused** on actual runtime errors, not development/deployment issues

### Database Initialization and Seeding

**PR Review Comment from @Selnapenek:**
> if you go with this services: please use seed instead of this method. not necessary call to try insert 1 raw every time when estate is creating.

**Context:** This comment was made regarding the `initializeCounter()` method in `ScanBoxIdService.ts` that attempted to insert an initial counter row on every estate creation.

**Practice:** 
- **Use database seeding** instead of runtime initialization methods for initial data
- **Avoid runtime database setup** that runs on every operation
- **Initialize required data during deployment/migration** not during application runtime
- **Don't perform database setup operations** as part of business logic execution

### Prisma ORM Usage

**PR Review Comment from @Selnapenek:**
> please don't use raw sql here, it is a simple case that is covered by prisma. use prisma orm as it is tx.scan_box_id_counter.find({ where: {...} })

**Context:** This comment was made regarding the use of raw SQL (`$queryRaw`) in `ScanBoxIdService.ts` for a simple SELECT query that could be handled by Prisma ORM methods.

**Practice:** 
- **Use Prisma ORM methods** instead of raw SQL for simple database operations
- **Prefer `tx.modelName.find()`** over `tx.$queryRaw` when the operation is straightforward
- **Reserve raw SQL** for complex queries that cannot be expressed with Prisma ORM
- **Maintain consistency** with the codebase's ORM usage patterns

### Error Handling and Business Logic Impact

**PR Review Comment from @Selnapenek:**
> what we will do when counter is ended and we hit limits? does estate creation will be blocked?

**Context:** This comment was made regarding the `ScanBoxIdExhaustedError` that would be thrown when the counter reaches its maximum value (0xFFFFFFFF), questioning the business impact of this error.

**Practice:** 
- **Consider business impact** of error conditions and limits
- **Plan for edge cases** that could block core business functionality
- **Document error handling strategy** for limit scenarios
- **Ensure graceful degradation** or alternative approaches when limits are reached
- **Evaluate whether blocking operations** is acceptable or if alternative solutions are needed

### Prisma ORM Usage and Query Optimization

**PR Review Comment from @Selnapenek:**
> please don't use raw sql
> here is an example of usage to make an updates
> 
> ```typescript
> const updatedCounter = await prisma.scan_box_id_counter.updateMany({
>   where: {
>     id: 1,
>     counter: {
>       lt: MAX_COUNTER,
>     },
>   },
>   data: {
>     counter: {
>       increment: 1,
>     },
>   },
> });
> ```
> 
> Prisma's fluent API is more intuitive and easier to read compared to raw SQL queries.
> Prisma can optimize queries, potentially improving performance compared to manually written SQL.

**Context:** This comment was made regarding the use of raw SQL (`$queryRaw`) for updating the counter table, suggesting the use of Prisma's fluent API instead.

**Practice:**
- **Use Prisma's fluent API** instead of raw SQL for database operations
- **Prefer `updateMany()` with `increment`** over raw SQL for counter updates
- **Leverage Prisma's query optimization** capabilities
- **Use Prisma's type-safe query builders** for better maintainability
- **Reserve raw SQL** only for complex operations that cannot be expressed with Prisma ORM
- **Choose readability and maintainability** over manual SQL control

### Database-First Solutions and Simplicity

**PR Review Comment from @Selnapenek:**
> Overall, all of the comments above ^ doesn't make much sense except for educational purposes.
> 
> Too complicated solution, if I understood task correctly it's all can be done with database. In that case this PR must contain just few lines of code.
> 
> ```sql
> CREATE SEQUENCE scan_box_id_seq;
> 
> ALTER TABLE Estate
> ADD COLUMN scanBoxId CHAR(8) UNIQUE DEFAULT lpad(to_hex(nextval(scan_box_id_seq)), 8, '0');
> ```

**Context:** This comment was made after reviewing the complex atomic counter implementation, suggesting a much simpler database-native solution using PostgreSQL sequences.

**Practice:**
- **Prefer database-native solutions** over application-level complexity
- **Use PostgreSQL sequences** for auto-incrementing values when appropriate
- **Leverage database functions** like `lpad()` and `to_hex()` for formatting
- **Keep implementations simple** - if the database can handle it, let it
- **Avoid over-engineering** when database features can solve the problem elegantly
- **Consider the database as part of the solution** rather than just storage
- **Use `DEFAULT` values** to handle automatic generation at the database level
- **Prefer declarative solutions** over imperative application code

### Dead Code Removal and Maintenance

**PR Review Comment from @Selnapenek:**
> seems like with current changes this file is not used anymore, please remove it

**Context:** This comment was made regarding `src/utils/scanBoxIdGenerator.ts` after the implementation changed to use a different approach for generating scanBoxIds.

**Practice:**
- **Remove unused files** when implementation changes make them obsolete
- **Keep codebase clean** by eliminating dead code
- **Regularly audit for unused imports and files** during refactoring
- **Don't leave commented-out or unused code** in the repository
- **Update documentation** when removing functionality
- **Ensure all references are updated** before removing files
- **Use tools like `grep` or IDE features** to verify files are truly unused before deletion

### Database Transactions and Data Consistency

**PR Review Comment from @s1owjke:**
> It must be done in transaction

**Context:** This comment was made regarding the `scanBoxId` generation in `EstateService.ts`, emphasizing that the ID generation should be part of the same transaction as the estate creation.

**Practice:**
- **Use database transactions** for operations that must be atomic
- **Include ID generation in the same transaction** as the main operation
- **Ensure data consistency** by keeping related operations in a single transaction
- **Avoid separate transactions** for operations that should be atomic
- **Consider rollback scenarios** - if estate creation fails, the generated ID should also be rolled back
- **Use Prisma's `$transaction()`** for multi-step operations that must succeed or fail together
- **Don't generate IDs outside the transaction** if they're part of the main operation

### Simplifying ID Generation

**PR Review Comment from @s1owjke:**
> You can generate scanBoxId without any services

**Context:** This comment was made regarding the `scanBoxId` generation in `EstateService.ts`, suggesting that the ID generation can be handled more simply without creating a separate service.

**Practice:**
- **Avoid creating services for simple operations** that can be handled inline
- **Use database-native solutions** when possible instead of application services
- **Prefer direct database features** over custom service layers for basic functionality
- **Keep implementations simple** - don't over-architect simple operations
- **Consider if a service adds value** or just adds unnecessary complexity
- **Use database `DEFAULT` values** for automatic ID generation when appropriate
- **Reserve services for complex business logic** that requires multiple steps or external dependencies

### Alternative ID Generation Approaches

**PR Review Comment from @IgorSolovyoff:**
> I guess the logic might be more simple if we use cuid generator instead of hand writed scripts: https://www.npmjs.com/package/@paralleldrive/cuid2
> 
> If we have unique check for scanBoxId so we can just do
> 
> ```typescript
> import { init } from '@paralleldrive/cuid2';
> const generateScanBoxId = init({
>   // the length of the id
>   length: 8,
> });
> 
> const scanBoxId = generateScanBoxId();
> const estate = await prisma.estate.create({
>  data: {
>   ...restArgs,
>   scanBoxId,
> },
> ```
> 
> And if it fails just regenerate the id.
> 
> In that case we dont scan_box_id_counter table, ScanBoxIdService, scanBoxIdGenerator. So the whole PR will be few lines of code
> 
> Or as suggested @s1owjke
> 
> ```typescript
> export function generateScanBoxId() {
>   return Math.random().toString(36).substring(2, 10)
> }
> ```

**Context:** This comment was made suggesting alternative approaches to ID generation that would be simpler than the complex atomic counter implementation.

**Practice:**
- **Consider multiple approaches** before settling on a complex solution
- **Evaluate external libraries** like CUID2 for ID generation when appropriate
- **Use database unique constraints** to handle collision detection
- **Implement retry logic** for collision scenarios when using random generation
- **Prefer simpler implementations** over complex custom solutions
- **Consider the trade-offs** between different approaches (random vs sequential vs external libraries)
- **Use established libraries** for common problems like ID generation
- **Leverage database constraints** to handle uniqueness validation

### Code Clarity and Purpose Documentation

**PR Review Comment from @IgorSolovyoff:**
> not sure whats the purpose of this table and constraints

**Context:** This comment was made regarding the `scan_box_id_counter` table and its constraints in the migration file, questioning the purpose and necessity of this implementation.

**Practice:**
- **Document the purpose** of complex database structures
- **Explain the reasoning** behind custom tables and constraints
- **Add comments to migrations** explaining why certain structures are needed
- **Consider if complexity is justified** - ask "why do we need this?"
- **Evaluate alternatives** before implementing complex solutions
- **Question the necessity** of additional tables when simpler solutions exist
- **Use descriptive names** that clearly indicate the purpose
- **Avoid creating tables** without clear justification for their existence

### GraphQL Schema Synchronization and Frontend Deployment

**Issue from AE-1159 Deployment:**
> Error: Property 'scanBoxId' does not exist on type '{ __typename?: "Estate2" | undefined; ... }'

**Context:** This error occurred during alix-estate-manager deployment when the frontend tried to use a new `scanBoxId` field that was added to the backend GraphQL schema, but the frontend's generated types hadn't been updated to include this field.

**Practice:**
- **Always regenerate GraphQL types** after backend schema changes that add new fields
- **Use the correct API endpoint** for GraphQL code generation (e.g., QA environment endpoint when deploying to QA)
- **Run `yarn generate:gql-types`** with the appropriate `GRAPHQL_ENDPOINT` environment variable
- **Test TypeScript compilation** (`yarn typecheck`) before committing frontend changes
- **Ensure schema synchronization** between backend and frontend before deployment
- **Include GraphQL regeneration** as a mandatory step in the deployment process
- **Verify generated types** include all new fields before pushing changes
- **Use environment-specific endpoints** for code generation (e.g., `GRAPHQL_ENDPOINT=https://api.qa3.meetalix.io/graphql`)
- **Apply linting fixes** to generated files before committing (`yarn lint:fix`)
- **Document the regeneration process** in deployment procedures

## Best Practices

*This section will contain general coding best practices identified from reviews.*

## Architecture Guidelines

*This section will contain architectural decisions and patterns from code reviews.*

---

*Last updated: [Date will be added when content is populated]*
