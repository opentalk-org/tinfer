# Coding Rules

## File and Folder Structure

* Keep each file under 300 lines.
* Keep each folder under 16 files.
* If a file or folder exceeds these limits, split the code into smaller modules.
* Prefer the single responsibility principle. When code does too many things, split it into focused units.

## Imports

* Always place imports at the top of the module.
* Avoid inline imports inside function bodies, type annotations, or interface fields.
* Inline imports are allowed only when there is a strict circular-dependency reason, and that reason must be documented.

## Project Assumptions

* This is a greenfield project.
* Do not add legacy-code support, compatibility fallbacks, or migration paths unless explicitly requested.

# Avoid

* Avoid 1–3 line functions unless they meaningfully improve readability or express intent.
* Avoid nested class or function definitions.
* Avoid `.get()` when direct indexing with `[]` is appropriate.
* Do not use default values or fallback behavior unless explicitly requested or clearly justified.
* Avoid raw dictionaries and strings for structured data. Prefer dataclasses, Pydantic models, and enums.
* Avoid excessive checking. Prefer clear failures over silent defaults, silent skips, or hidden fallback behavior.
* Do not overuse `if` statements.
* Do not repeatedly check values that already have defaults or guaranteed invariants.

# If Statements

Use `if` statements only when both branches are valid, expected paths.

## Wrong

```rust
if condition {
    // happy path
} else {
    // "shouldn't happen" - silently ignored
}
```

## Right

```rust
assert!(condition, "invariant violated: ...");
```

Or:

```rust
return Err(LimboError::InternalError("unexpected state".into()));
```

Or:

```rust
unreachable!("impossible state: ...");
```

If only one branch should ever be reached, use an assertion, explicit error, or `unreachable!` instead of silently ignoring the impossible branch.

# Comments and Documentation

## Do

* Document why something exists, not what the code does.
* Document functions, structs, enums, and enum variants when useful.
* Explain why a decision, constraint, or workaround is necessary.

## Don’t

* Do not write comments that simply repeat the code.
* Do not reference AI conversations or prompts.
* Do not use temporal markers such as “added,” “existing code,” “new,” or “Phase 1.”
* Do not add comments or docstrings to unchanged code unless requested.

# Avoid Over-Engineering

* Make only the changes directly requested or clearly necessary.
* Do not add features beyond the request.
* Do not add error handling for impossible scenarios.
* Do not create abstractions for one-time operations.
* Prefer three similar lines over premature abstraction.