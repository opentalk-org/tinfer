# Phoneme Duration Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one reproducible command that embeds 20 seeded-random reference WAVs, generates 320 unchunked Polish utterances, and produces predictor-duration metrics globally and for four voices.

**Architecture:** A typed data module owns the fixed corpus and metric records, an inference module owns archive/model operations, a reporting module owns aggregation and plots, and a small runner orchestrates the stages. Production inference remains unchanged; phoneme alignments expose the exact rounded predictor frame allocation used by synthesis.

**Tech Stack:** Python 3.11, PyTorch, StyleTTS2, NumPy, soundfile, matplotlib, tqdm, unittest.

## Global Constraints

- Run with one command from `/workspace/tinfer`.
- Select exactly 20 reference WAVs and four highlighted voices with seed `20260710`.
- Generate 16 fixed, real Polish inputs spanning 2–300 characters for every voice.
- Call `StyleTTS2.generate()` directly and reject internal model-window splitting.
- Use predictor duration for both per-phoneme duration and phonemes/second.
- Keep every file below 300 lines and the `test_speed` folder below 16 files.
- Preserve unrelated worktree changes.

---

### Task 1: Typed Corpus and Metric Primitives

**Files:**
- Create: `test_speed/__init__.py`
- Create: `test_speed/benchmark_data.py`
- Test: `test_speed/test_benchmark.py`

**Interfaces:**
- Produces: `BenchmarkConfig`, `TextInput`, `RequestMetric`, `PhonemeMetric`, `SummaryRow`, `POLISH_INPUTS`, `select_names()`, `summarize_phonemes()`.
- Consumes: Python dataclasses, `random.Random`, and NumPy percentile calculations.

- [ ] **Step 1: Write failing selection and aggregation tests**

```python
class DataTests(unittest.TestCase):
    def test_selection_is_seeded_and_sorted(self) -> None:
        selected = select_names([f"v{i}" for i in range(30)], 20, 7)
        self.assertEqual(selected, select_names([f"v{i}" for i in range(30)], 20, 7))
        self.assertEqual(selected, sorted(selected))

    def test_summary_groups_each_phoneme(self) -> None:
        rows = summarize_phonemes([
            PhonemeMetric("a", 0.025, "v", "t"),
            PhonemeMetric("a", 0.075, "v", "t"),
        ])
        self.assertEqual(rows[0].count, 2)
        self.assertAlmostEqual(rows[0].average_seconds, 0.05)
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark.DataTests -v`

Expected: FAIL because `test_speed.benchmark_data` does not exist.

- [ ] **Step 3: Implement typed records, corpus, selection, and statistics**

```python
@dataclass(frozen=True)
class PhonemeMetric:
    phoneme: str
    duration_seconds: float
    voice_id: str
    text_id: str

def select_names(names: list[str], count: int, seed: int) -> list[str]:
    if len(names) < count:
        raise ValueError(f"Need {count} inputs, found {len(names)}")
    return sorted(random.Random(seed).sample(sorted(names), count))

def summarize_phonemes(metrics: list[PhonemeMetric]) -> list[SummaryRow]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.phoneme].append(metric.duration_seconds)
    return [SummaryRow.from_values(symbol, grouped[symbol]) for symbol in sorted(grouped)]
```

Define all six dataclasses with explicit fields and place 16 Polish `TextInput` constants at measured lengths from 2 through 300 characters.

- [ ] **Step 4: Run data tests**

Run the command from Step 2. Expected: all `DataTests` pass.

- [ ] **Step 5: Commit the data layer**

```bash
git add test_speed/__init__.py test_speed/benchmark_data.py test_speed/test_benchmark.py
git commit -m "feat: add phoneme benchmark data model"
```

### Task 2: Archive, Embedding, and Unchunked Synthesis

**Files:**
- Create: `test_speed/benchmark_inference.py`
- Modify: `test_speed/test_benchmark.py`

**Interfaces:**
- Consumes: `BenchmarkConfig`, `TextInput`, `RequestMetric`, `PhonemeMetric`.
- Produces: `archive_wav_names()`, `extract_selected()`, `decoder_frame_seconds()`, `measure_result()`, `load_model()`, `embed_references()`, `synthesize_all()`.

- [ ] **Step 1: Write failing extraction and duration tests**

```python
class InferenceTests(unittest.TestCase):
    def test_measure_result_uses_predictor_alignment(self) -> None:
        metadata = {"word_alignments": [AlignmentItem("a", 0, 1, 0, 25)]}
        result = SimpleNamespace(metadata=metadata)
        request, phonemes = measure_result(result, "v", TextInput("short", "No"), Path("a.wav"))
        self.assertEqual(request.phoneme_count, 1)
        self.assertAlmostEqual(request.phonemes_per_second, 40.0)
        self.assertAlmostEqual(phonemes[0].duration_seconds, 0.025)

    def test_merged_model_windows_are_rejected(self) -> None:
        result = SimpleNamespace(metadata={"window_count": 2, "word_alignments": []})
        with self.assertRaisesRegex(RuntimeError, "chunked"):
            measure_result(result, "v", TextInput("short", "No"), Path("a.wav"))
```

- [ ] **Step 2: Run tests and verify missing-function failure**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark.InferenceTests -v`

Expected: FAIL because inference functions do not exist.

- [ ] **Step 3: Implement safe extraction and same-pass measurement**

```python
def measure_result(result: IntermediateRepresentation, voice_id: str, text: TextInput, audio_path: Path) -> tuple[RequestMetric, list[PhonemeMetric]]:
    if "window_count" in result.metadata:
        raise RuntimeError(f"Model internally chunked {text.text_id}")
    alignments = result.metadata["word_alignments"]
    if not alignments:
        raise RuntimeError(f"No predictor alignments for {text.text_id}")
    phonemes = [PhonemeMetric(item.item, (item.end_ms - item.start_ms) / 1000.0, voice_id, text.text_id) for item in alignments]
    predicted_seconds = sum(item.duration_seconds for item in phonemes)
    request = RequestMetric(voice_id, text.text_id, text.text, len(text.text), len(phonemes), predicted_seconds, len(phonemes) / predicted_seconds, str(audio_path))
    return request, phonemes
```

`extract_selected()` must use `ZipFile.open()` plus `shutil.copyfileobj()` and reject names outside the selected basename set. `embed_references()` calls the loaded model’s voice encoder and saves each tensor as `{"voice_vector": tensor.cpu()}`. `synthesize_all()` uses direct `model.generate()` with `AlignmentType.PHONEME`, writes float WAVs, and returns typed metrics.

- [ ] **Step 4: Run inference tests**

Run the command from Step 2. Expected: all `InferenceTests` pass.

- [ ] **Step 5: Commit inference operations**

```bash
git add test_speed/benchmark_inference.py test_speed/test_benchmark.py
git commit -m "feat: add unchunked duration benchmark inference"
```

### Task 3: Reports and One-Command Runner

**Files:**
- Create: `test_speed/benchmark_reporting.py`
- Create: `test_speed/run_benchmark.py`
- Modify: `test_speed/test_benchmark.py`

**Interfaces:**
- Consumes: typed request, phoneme, and summary records plus inference-stage functions.
- Produces: `write_raw_metrics()`, `write_summary_table()`, `plot_scatter()`, `write_summary_index()`, and `main()`.

- [ ] **Step 1: Write failing report artifact test**

```python
class ReportingTests(unittest.TestCase):
    def test_report_writes_global_and_voice_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = [RequestMetric("v", "short", "No", 2, 1, 0.025, 40.0, "a.wav")]
            phonemes = [PhonemeMetric("a", 0.025, "v", "short")]
            write_reports(root, requests, phonemes, ["v"])
            self.assertTrue((root / "summary/global_phoneme_durations.csv").is_file())
            self.assertTrue((root / "summary/global_phonemes_per_second.png").is_file())
            self.assertTrue((root / "summary/v_phoneme_durations.csv").is_file())
```

- [ ] **Step 2: Run test and verify missing-function failure**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark.ReportingTests -v`

Expected: FAIL because reporting functions do not exist.

- [ ] **Step 3: Implement CSV/JSON, plots, manifest, and runner**

```python
def plot_scatter(metrics: list[RequestMetric], title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.scatter([row.text_length for row in metrics], [row.phonemes_per_second for row in metrics], alpha=0.65)
    axis.set(title=title, xlabel="Input text length (characters)", ylabel="Predicted phonemes/s")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
```

The runner seeds Python, NumPy, and PyTorch; recreates benchmark-owned output directories; selects 20 and then four names; loads the model once; runs extraction, embedding, synthesis, and reporting; and writes `manifest.json`. Each long stage receives a `tqdm` iterable. Its module entry point is guarded by `if __name__ == "__main__": main()`.

- [ ] **Step 4: Run all unit tests**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark -v`

Expected: all tests pass.

- [ ] **Step 5: Run the full single command**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m test_speed.run_benchmark`

Expected: four `tqdm` stages complete; 20 embeddings and 320 non-empty WAVs exist; global and four voice tables/scatters exist; the script prints the summary index path.

- [ ] **Step 6: Validate artifact counts and report schemas**

Run: `find test_speed/results/embeddings -name '*.pth' | wc -l && find test_speed/results/audio -name '*.wav' | wc -l && find test_speed/results/summary -name '*phoneme_durations.csv' | wc -l && find test_speed/results/summary -name '*phonemes_per_second.png' | wc -l`

Expected lines: `20`, `320`, `5`, `5`.

- [ ] **Step 7: Commit the completed benchmark**

```bash
git add test_speed/benchmark_reporting.py test_speed/run_benchmark.py test_speed/test_benchmark.py
git commit -m "feat: add reproducible phoneme duration benchmark"
```
