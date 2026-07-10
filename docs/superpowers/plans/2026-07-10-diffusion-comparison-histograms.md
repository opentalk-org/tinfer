# Diffusion Comparison Histograms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate diffusion-enabled and no-diffusion benchmark folders in one command, with per-voice and all-run phonemes/s histograms in both.

**Architecture:** Parameterize synthesis with an explicit `use_diffusion` flag and let the runner execute two named profiles against one loaded model and one set of embedded references. Compute histogram bins from both profiles together, then pass those shared edges into the existing report writer.

**Tech Stack:** Python 3.11, PyTorch, StyleTTS2, NumPy, matplotlib, soundfile, tqdm, unittest.

## Global Constraints

- Preserve `test_speed/results/` as the diffusion-enabled profile.
- Create `test_speed/results_no_diffusion/` with `use_diffusion=False`.
- Use the same 20 voices, four highlighted voices, and 16 Polish inputs in both profiles.
- Extract and embed once, then copy references and serialized embeddings into the second folder.
- Render one 20-value voice-average histogram and one 320-value all-run histogram per profile.
- Use shared 1 phoneme/s-wide bin edges across corresponding profile plots.
- Keep every file below 300 lines and `test_speed/` below 16 files.
- Preserve unrelated worktree and staged changes.

---

### Task 1: Parameterized Diffusion Synthesis

**Files:**
- Modify: `test_speed/benchmark_data.py`
- Modify: `test_speed/benchmark_inference.py`
- Modify: `test_speed/test_benchmark.py`

**Interfaces:**
- Produces: `SynthesisProfile(name: str, results_dir: Path, use_diffusion: bool)`.
- Changes: `synthesize_all(model, voice_ids, text_inputs, output_dir, use_diffusion)` forwards the flag in every model parameter dictionary.

- [ ] **Step 1: Write a failing flag-forwarding test**

```python
class RecordingModel:
    _max_styletts_tokens = 512

    def __init__(self) -> None:
        self.params: list[dict[str, object]] = []

    def _text_token_count(self, text: str, params: StyleTTS2Params) -> int:
        return len(text)

    def generate(self, text, context, params, metadata):
        self.params.append(params)
        return SimpleNamespace(
            data=np.zeros(240, dtype=np.float32),
            sample_rate=24000,
            metadata={"word_alignments": [AlignmentItem("n", 0, 1, 0, 25)]},
        )

def test_synthesis_forwards_diffusion_flag(self) -> None:
    model = RecordingModel()
    with tempfile.TemporaryDirectory() as directory:
        synthesize_all(model, ["voice"], [TextInput("short", "No")], Path(directory), False)
    self.assertEqual(model.params, [{"use_diffusion": False}])
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark.InferenceTests.test_synthesis_forwards_diffusion_flag -v`

Expected: FAIL because `synthesize_all` has no `use_diffusion` argument.

- [ ] **Step 3: Add the typed profile and forward the explicit parameter**

```python
@dataclass(frozen=True)
class SynthesisProfile:
    name: str
    results_dir: Path
    use_diffusion: bool

def synthesize_all(model, voice_ids, text_inputs, output_dir, use_diffusion):
    result = model.generate(
        text_input.text,
        {"voice_id": voice_id},
        {"use_diffusion": use_diffusion},
        {"alignment_type": AlignmentType.PHONEME},
    )
```

- [ ] **Step 4: Run all benchmark tests**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark -v`

Expected: all tests pass.

- [ ] **Step 5: Commit parameterized synthesis**

```bash
git add test_speed/benchmark_data.py test_speed/benchmark_inference.py test_speed/test_benchmark.py
git commit -m "feat: parameterize benchmark diffusion"
```

### Task 2: Shared Histograms and Two-Profile Runner

**Files:**
- Modify: `test_speed/.gitignore`
- Modify: `test_speed/benchmark_reporting.py`
- Modify: `test_speed/run_benchmark.py`
- Modify: `test_speed/test_benchmark.py`

**Interfaces:**
- Produces: `mean_rates_by_voice() -> list[float]`, `shared_histogram_edges() -> np.ndarray`, and `plot_histogram()`.
- Changes: `write_reports(..., histogram_edges)` writes both histogram PNGs; `main()` prepares and synthesizes both `SynthesisProfile` values.

- [ ] **Step 1: Write failing aggregation and artifact tests**

```python
def test_voice_histogram_values_average_each_voice(self) -> None:
    requests = [
        RequestMetric("a", "x", "x", 1, 1, 0.05, 20.0, "a.wav"),
        RequestMetric("a", "y", "y", 1, 1, 0.025, 40.0, "b.wav"),
        RequestMetric("b", "x", "x", 1, 1, 0.025, 40.0, "c.wav"),
    ]
    self.assertEqual(mean_rates_by_voice(requests), [30.0, 40.0])

def test_report_writes_both_histograms(self) -> None:
    write_reports(root, requests, phonemes, ["v"], np.arange(39.0, 42.0, 1.0))
    self.assertTrue((root / "summary/phonemes_per_second_by_voice.png").is_file())
    self.assertTrue((root / "summary/phonemes_per_second_all_runs.png").is_file())
```

- [ ] **Step 2: Run reporting tests and verify RED**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark.ReportingTests -v`

Expected: FAIL because the histogram helpers and new report signature do not exist.

- [ ] **Step 3: Implement shared bins and histograms**

```python
def mean_rates_by_voice(metrics: list[RequestMetric]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.voice_id].append(metric.phonemes_per_second)
    return [float(np.mean(grouped[voice_id])) for voice_id in sorted(grouped)]

def shared_histogram_edges(*groups: list[float]) -> np.ndarray:
    values = np.asarray([value for group in groups for value in group])
    lower = np.floor(values.min())
    upper = np.ceil(values.max())
    return np.arange(lower, upper + 1.0, 1.0)
```

`plot_histogram()` calls `axis.hist(values, bins=edges)`, labels the x-axis `Predicted phonemes/s`, labels the y-axis with the population name, and asserts the histogram count sum equals `len(values)`.

- [ ] **Step 4: Implement two-profile orchestration**

```python
profiles = [
    SynthesisProfile("diffusion", ROOT / "test_speed/results", True),
    SynthesisProfile("no_diffusion", ROOT / "test_speed/results_no_diffusion", False),
]
```

Prepare both folders, extract/embed into the first, copy its `references/` and `embeddings/` trees into the second with `shutil.copytree(source, destination, dirs_exist_ok=True)`, synthesize both profiles, compute shared edges from both request populations, write both reports, and write profile name plus `use_diffusion` to each manifest. Add `results_no_diffusion/` to `test_speed/.gitignore`.

- [ ] **Step 5: Run all unit tests**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark -v`

Expected: all tests pass.

- [ ] **Step 6: Run and audit the full command**

Run: `./test_speed/run_benchmark.sh`

Expected: both synthesis progress bars reach 320; each folder contains 20 embeddings, 320 non-empty WAVs, five duration tables, five scatter plots, and two histogram plots.

- [ ] **Step 7: Commit and verify**

```bash
git add test_speed/.gitignore test_speed/benchmark_reporting.py test_speed/run_benchmark.py test_speed/test_benchmark.py
git commit -m "feat: compare diffusion benchmark profiles"
```

Run the unit tests and artifact-count audit once more after committing.
