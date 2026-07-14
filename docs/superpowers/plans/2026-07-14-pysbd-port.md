# Native pySBD Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Rust `pysbd` crate matching every behavior and copied test from upstream commit `5905f13`.

**Architecture:** A direct shared processor operates on centralized rule structs. Every language remains a separate
module and test module; special behavior is selected explicitly by `Language`, without a trait hierarchy or Python.

**Tech Stack:** Rust 2024, `fancy-regex`, `regex`, upstream pySBD MIT sources and test corpus.

## Global Constraints

- Pin behavior to pySBD commit `5905f13be4fc95f407b98392e0ec303617a33d86`.
- Full surface: cleaning, PDF mode, non-destructive segmentation, spans, all regressions, and 23 languages.
- Keep files under 300 lines and folders under 16 files.
- One Rust source and one Rust test module per language.
- No Python runtime, build-time generator, legacy fallback, or hidden skip.
- Copy upstream MIT license and attribution.

---

### Task 1: Crate contract and shared rule engine

**Files:**
- Create: `tinfer_rust/pysbd/Cargo.toml`, `LICENSE`, `NOTICE`
- Create: `tinfer_rust/pysbd/src/{lib,error,rules,text}.rs`
- Create: `tinfer_rust/pysbd/tests/{segmenter,languages}.rs`

**Interfaces:**
- Produces: `Options`, `DocType`, `TextSpan`, `Segmenter`, `Language`, `Rules`, and crate `Result<T>`.

- [ ] Write contract tests for empty input, unsupported language, option conflicts, whitespace-preserving spans,
and `join(segment(text)) == text` using upstream `tests/test_segmenter.py` inputs.
- [ ] Run `cargo test --manifest-path tinfer_rust/pysbd/Cargo.toml`; expect unresolved contract failures.
- [ ] Implement the exact public declarations:

```rust
pub struct Options { pub clean: bool, pub doc_type: Option<DocType> }
pub enum DocType { Pdf }
pub struct TextSpan { pub text: String, pub start: usize, pub end: usize }
pub struct Segmenter { language: Language, options: Options }
impl Segmenter {
    pub fn new(language: &str, options: Options) -> Result<Self>;
    pub fn segment(&self, text: &str) -> Result<Vec<String>>;
    pub fn segment_spans(&self, text: &str) -> Result<Vec<TextSpan>>;
}
```

- [ ] Implement `Rule { pattern, replacement }`, ordered replacement helpers, Python replacement-group conversion,
and UTF-8 span lookup. Compile unsupported patterns with `fancy_regex::Regex` and return errors.
- [ ] Run crate tests; expect contract tests to pass and processing cases to remain absent.
- [ ] Commit `feat(pysbd): add standalone crate contract`.

### Task 2: Shared processor and cleaner

**Files:**
- Create: `tinfer_rust/pysbd/src/{processor,cleaner,abbreviation,punctuation,lists}.rs`
- Create: `tinfer_rust/pysbd/tests/{cleaner,regression,english_clean}.rs`

**Interfaces:**
- Consumes: Task 1 `Rules`, ordered replacement helpers, `Language`.
- Produces: `processor::process(text, language)` and `cleaner::clean(text, language, doc_type)`.

- [ ] Translate every row from upstream `test_cleaner.py`, `test_english_clean.py`, and `tests/regression` into
table-driven Rust tests with verbatim inputs/expected outputs.
- [ ] Run the three test targets; expect processing mismatches.
- [ ] Port, in upstream call order, list line breaks, abbreviation replacement, number protection, continuous
punctuation, numbered references, email/file/location protection, sentence boundaries, symbol restoration, and
non-destructive span recovery.
- [ ] Port cleaner rules in upstream order, including HTML removal, PDF newline joining, bullet/list handling, and
whitespace normalization. Preserve input ownership and empty-input behavior.
- [ ] Run `cargo test --manifest-path tinfer_rust/pysbd/Cargo.toml --test cleaner --test regression --test english_clean`;
expect zero failures.
- [ ] Commit `feat(pysbd): port shared processor and cleaner`.

### Task 3: Common, English, and standard rules

**Files:**
- Create: `tinfer_rust/pysbd/src/lang/{mod,common,standard}.rs`
- Create: `tinfer_rust/pysbd/src/lang/a_f/english.rs`
- Create: `tinfer_rust/pysbd/tests/lang/a_f/english.rs`

**Interfaces:**
- Produces: `Language::English` rules and common defaults used by every language task.

- [ ] Copy every English segmentation assertion from upstream `test_english.py` into the Rust module.
- [ ] Port `lang/common/common.py`, `lang/common/standard.py`, and `lang/english.py` constants and ordered lists.
- [ ] Run English, segmenter, and regression tests; fix only parity defects in shared behavior.
- [ ] Commit `feat(pysbd): port common and English rules`.

### Task 4: Languages A through F

**Files:**
- Create one file each under `tinfer_rust/pysbd/src/lang/a_f/`: `amharic.rs`, `arabic.rs`, `armenian.rs`,
`bulgarian.rs`, `burmese.rs`, `chinese.rs`, `danish.rs`, `dutch.rs`, `french.rs`.
- Create matching files under `tinfer_rust/pysbd/tests/lang/a_f/`.

**Interfaces:**
- Consumes: Task 3 common `Rules` constructor.
- Produces: exact `Language` match arms and rules for codes `am ar hy bg my zh da nl fr`.

- [ ] For each language, translate its upstream test module verbatim before implementation and confirm failure.
- [ ] Copy its upstream language constants, abbreviations, sentence starters, and custom hooks into only that
language's Rust file; add the exact ISO-code match arm.
- [ ] Run that language test target and shared regression tests; require zero failures before its commit.
- [ ] Commit each language independently as `feat(pysbd): port <language> rules`.

### Task 5: Languages G through L

**Files:**
- Create one file each under `tinfer_rust/pysbd/src/lang/g_l/`: `german.rs`, `greek.rs`, `hindi.rs`, `italian.rs`,
`japanese.rs`, `kazakh.rs`.
- Create matching files under `tinfer_rust/pysbd/tests/lang/g_l/`.

**Interfaces:**
- Consumes: Task 3 common `Rules` constructor.
- Produces: exact `Language` match arms and rules for codes `de el hi it ja kk`.

- [ ] For each language, translate its upstream test module verbatim before implementation and confirm failure.
- [ ] Copy its upstream language constants, abbreviations, sentence starters, and custom hooks into only that file.
- [ ] Run its test target and shared regressions; require zero failures before its independent language commit.

### Task 6: Languages M through Z

**Files:**
- Create one file each under `tinfer_rust/pysbd/src/lang/m_z/`: `marathi.rs`, `persian.rs`, `polish.rs`,
`russian.rs`, `slovak.rs`, `spanish.rs`, `urdu.rs`.
- Create matching files under `tinfer_rust/pysbd/tests/lang/m_z/`.

**Interfaces:**
- Consumes: Task 3 common `Rules` constructor.
- Produces: exact `Language` match arms and rules for codes `mr fa pl ru sk es ur`.

- [ ] For each language, translate its upstream test module verbatim before implementation and confirm failure.
- [ ] Copy its upstream language constants, abbreviations, sentence starters, and custom hooks into only that file.
- [ ] Run its test target and shared regressions; require zero failures before its independent language commit.

### Task 7: Whole-crate audit

**Files:**
- Modify only parity defects found in `tinfer_rust/pysbd/src/**` and `tests/**`.

**Interfaces:**
- Produces: accepted standalone crate for tinfer integration.

- [ ] Compare upstream test function/parameter counts with Rust case counts and fail if any upstream row is absent.
- [ ] Run `cargo fmt --manifest-path tinfer_rust/pysbd/Cargo.toml --check` and full crate tests; expect zero failures.
- [ ] Run Clippy with `-D warnings`; resolve findings without changing behavior.
- [ ] Verify every source/test file is under 300 lines and every folder under 16 files.
- [ ] Commit `test(pysbd): prove full upstream parity`.
