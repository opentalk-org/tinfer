# Maximum Phoneme Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 16 character-length inputs with 48 natural Polish word-boundary prefixes spanning the model's phoneme-token window through the largest prefix fitting within 511 tokens.

**Architecture:** A focused corpus module owns the Polish passage and converts its word-boundary prefixes into a deterministic tokenizer-measured grid. Typed request data carries the measured input token count through synthesis and reporting, while the runner builds the grid after model loading and uses it for both profiles.

**Tech Stack:** Python 3.11, StyleTTS2 phonemizer/tokenizer, NumPy, unittest.

## Global Constraints

- Produce exactly 48 unique, strictly increasing token counts.
- Count model phoneme tokens excluding BOS.
- Keep every input at or below 511 tokens.
- Make the final input the longest natural word-boundary prefix fitting at or below 511.
- Use input phoneme tokens on every scatter x-axis.
- Keep speed correction disabled through the existing benchmark monkeypatch.
- Generate 960 WAVs per profile and 1,920 total.
- Do not modify any file under `tinfer/`.
- Keep every file below 300 lines and `test_speed/` below 16 files.

---

### Task 1: Token-Measured Polish Corpus Grid

**Files:**
- Create: `test_speed/benchmark_corpus.py`
- Modify: `test_speed/benchmark_data.py`
- Modify: `test_speed/test_benchmark_profiles.py`

**Interfaces:**
- Changes: `TextInput(text_id: str, text: str, input_phoneme_tokens: int)`.
- Produces: `POLISH_PASSAGE` and `build_phoneme_grid(model, passage, point_count, max_tokens) -> list[TextInput]`.

- [ ] **Step 1: Write a failing grid test with a deterministic fake tokenizer**

```python
class WordTokenModel:
    def _text_token_count(self, text: str, params: StyleTTS2Params) -> int:
        return len(text.split()) * 2 + 1

def test_grid_has_48_increasing_points_and_maximum_prefix(self) -> None:
    passage = " ".join(f"word{i}" for i in range(80))
    grid = build_phoneme_grid(WordTokenModel(), passage, 48, 100)
    counts = [item.input_phoneme_tokens for item in grid]
    self.assertEqual(len(grid), 48)
    self.assertEqual(counts, sorted(set(counts)))
    self.assertEqual(counts[-1], 100)
    self.assertEqual(len(grid[-1].text.split()), 50)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark_profiles.PhonemeGridTests -v`

Expected: FAIL because `test_speed.benchmark_corpus` does not exist.

- [ ] **Step 3: Implement prefix candidates and target selection**

```python
@dataclass(frozen=True)
class PrefixCandidate:
    text: str
    token_count: int

def build_phoneme_grid(model, passage, point_count, max_tokens):
    words = passage.split()
    candidates = []
    for word_count in range(1, len(words) + 1):
        text = " ".join(words[:word_count])
        token_count = model._text_token_count(text, StyleTTS2Params()) - 1
        candidates.append(PrefixCandidate(text, token_count))
    fitting = [item for item in candidates if item.token_count <= max_tokens]
    assert len(fitting) >= point_count
    assert len(fitting) < len(candidates)
    targets = np.linspace(fitting[0].token_count, fitting[-1].token_count, point_count)
    selected = [max((item for item in fitting if item.token_count <= target), key=lambda item: item.token_count) for target in targets]
    assert len({item.token_count for item in selected}) == point_count
    return [TextInput(f"phonemes_{item.token_count:03d}", item.text, item.token_count) for item in selected]
```

Store a curated Polish passage long enough to exceed 511 model tokens.

- [ ] **Step 4: Run grid tests and commit**

Run the focused test. Expected: PASS.

```bash
git add test_speed/benchmark_corpus.py test_speed/benchmark_data.py test_speed/test_benchmark_profiles.py
git commit -m "feat: build maximum phoneme input grid"
```

### Task 2: Carry Token Counts into Metrics and Scatters

**Files:**
- Modify: `test_speed/benchmark_data.py`
- Modify: `test_speed/benchmark_inference.py`
- Modify: `test_speed/benchmark_reporting.py`
- Modify: `test_speed/test_benchmark.py`
- Modify: `test_speed/test_benchmark_profiles.py`

**Interfaces:**
- Changes: `RequestMetric` gains `input_phoneme_tokens: int` after `text_length`.
- Produces: `scatter_coordinates(metrics) -> tuple[list[int], list[float]]`.

- [ ] **Step 1: Write failing propagation and scatter tests**

```python
def test_result_preserves_input_phoneme_tokens(self) -> None:
    text = TextInput("phonemes_007", "tekst", 7)
    request, _ = measure_result(result, "voice", text, Path("a.wav"))
    self.assertEqual(request.input_phoneme_tokens, 7)

def test_scatter_uses_input_phoneme_tokens(self) -> None:
    metrics = [RequestMetric("v", "t", "x", 1, 7, 1, 0.05, 20.0, "a.wav")]
    self.assertEqual(scatter_coordinates(metrics), ([7], [20.0]))
```

- [ ] **Step 2: Run focused tests and verify RED**

Run the new test methods. Expected: FAIL because the metric field and helper do not exist.

- [ ] **Step 3: Implement propagation and scatter coordinates**

```python
request = RequestMetric(
    voice_id=voice_id,
    text_id=text_input.text_id,
    text=text_input.text,
    text_length=len(text_input.text),
    input_phoneme_tokens=text_input.input_phoneme_tokens,
    phoneme_count=len(phonemes),
    predicted_seconds=predicted_seconds,
    phonemes_per_second=len(phonemes) / predicted_seconds,
    audio_path=str(audio_path),
)

def scatter_coordinates(metrics):
    return (
        [row.input_phoneme_tokens for row in metrics],
        [row.phonemes_per_second for row in metrics],
    )
```

Use the helper in `plot_scatter()` and label the x-axis `Input phoneme tokens`.

- [ ] **Step 4: Update constructors, run all tests, and commit**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark test_speed.test_benchmark_profiles -v`

Expected: all tests pass.

```bash
git add test_speed/benchmark_data.py test_speed/benchmark_inference.py test_speed/benchmark_reporting.py test_speed/test_benchmark.py test_speed/test_benchmark_profiles.py
git commit -m "feat: report input phoneme token counts"
```

### Task 3: Two-Profile 48-Point Regeneration

**Files:**
- Modify: `test_speed/run_benchmark.py`
- Modify: `test_speed/test_benchmark.py`

**Interfaces:**
- Consumes: `build_phoneme_grid(model, POLISH_PASSAGE, 48, 511)`.
- Changes: validation and manifests use the runtime `text_inputs` list rather than `POLISH_INPUTS`.

- [ ] **Step 1: Replace static corpus orchestration**

```python
text_inputs = build_phoneme_grid(
    model,
    POLISH_PASSAGE,
    point_count=48,
    max_tokens=511,
)
```

Pass `text_inputs` into both synthesis calls, profile validation, and manifest serialization. Remove `POLISH_INPUTS` and its 16-input tests.

- [ ] **Step 2: Run tests and full benchmark**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark test_speed.test_benchmark_profiles -v`

Run: `./test_speed/run_benchmark.sh`

Expected: both profiles reach 960 requests; manifests contain 48 increasing input counts ending at the largest natural prefix at or below 511; all 1,920 WAVs are non-empty.

- [ ] **Step 3: Audit and commit**

Verify both profiles contain 960 request rows, 48 requests per voice, scatter plots use the expanded token range, and no result contains window-split metadata.

```bash
git add test_speed/run_benchmark.py test_speed/test_benchmark.py
git commit -m "feat: benchmark full phoneme context window"
```
