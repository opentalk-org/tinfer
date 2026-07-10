# Uncorrected Speed and Fine Histograms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Disable StyleTTS2 length-based speed correction only inside the benchmark and regenerate both profiles with 0.25 phoneme/s histogram bins.

**Architecture:** The benchmark runner monkeypatches the correction symbol imported by the StyleTTS2 model module with an identity function before synthesis. Histogram edge generation receives an explicit width, and each manifest records that speed correction was disabled.

**Tech Stack:** Python 3.11, StyleTTS2, NumPy, matplotlib, unittest.

## Global Constraints

- Do not modify any file under `tinfer/`.
- Apply the monkeypatch to both diffusion and no-diffusion profiles.
- Preserve requested `speed=1.0` for every text length.
- Use shared 0.25 phoneme/s-wide histogram bins.
- Record `speed_correction: false` in both manifests.
- Preserve unrelated worktree and staged changes.

---

### Task 1: Benchmark Speed-Correction Monkeypatch

**Files:**
- Modify: `test_speed/run_benchmark.py`
- Modify: `test_speed/test_benchmark.py`

**Interfaces:**
- Produces: `preserve_requested_speed(speed: float, token_count: int) -> float` and `disable_speed_correction() -> None`.
- Changes: `_write_manifest()` emits `speed_correction: false`; `main()` patches before any synthesis.

- [ ] **Step 1: Write the failing monkeypatch test**

```python
def test_speed_correction_is_disabled_for_benchmark(self) -> None:
    original = styletts2_model_module.baseline_speed_corrected_for_request
    try:
        disable_speed_correction()
        corrected = styletts2_model_module.baseline_speed_corrected_for_request(1.0, 300)
        self.assertEqual(corrected, 1.0)
    finally:
        styletts2_model_module.baseline_speed_corrected_for_request = original
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark.RunnerTests.test_speed_correction_is_disabled_for_benchmark -v`

Expected: FAIL because `disable_speed_correction` does not exist.

- [ ] **Step 3: Implement the benchmark-only patch**

```python
def preserve_requested_speed(speed: float, token_count: int) -> float:
    return speed

def disable_speed_correction() -> None:
    styletts2_model_module.baseline_speed_corrected_for_request = preserve_requested_speed
```

Call `disable_speed_correction()` in `main()` before synthesis and add `"speed_correction": False` to each manifest.

- [ ] **Step 4: Run all benchmark tests**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark -v`

Expected: all tests pass.

### Task 2: Quarter-Unit Histogram Bins and Regeneration

**Files:**
- Modify: `test_speed/benchmark_reporting.py`
- Modify: `test_speed/run_benchmark.py`
- Modify: `test_speed/test_benchmark.py`

**Interfaces:**
- Changes: `shared_histogram_edges(*groups: list[float], bin_width: float) -> np.ndarray` uses the explicit width.
- Produces: `HISTOGRAM_BIN_WIDTH = 0.25` in the runner.

- [ ] **Step 1: Change the edge test first and verify RED**

```python
edges = shared_histogram_edges([9.1], [10.1], bin_width=0.25)
np.testing.assert_array_equal(edges, np.arange(9.0, 10.5, 0.25))
```

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark.ReportingTests.test_shared_histogram_edges_cover_both_profiles -v`

Expected: FAIL because `bin_width` is not accepted.

- [ ] **Step 2: Implement explicit bin width**

```python
def shared_histogram_edges(*groups: list[float], bin_width: float) -> np.ndarray:
    values = np.asarray([value for group in groups for value in group])
    lower = np.floor(values.min() / bin_width) * bin_width
    upper = np.ceil(values.max() / bin_width) * bin_width
    upper = max(upper, lower + bin_width)
    return np.arange(lower, upper + bin_width, bin_width)
```

Pass `bin_width=HISTOGRAM_BIN_WIDTH` from the runner.

- [ ] **Step 3: Run tests and the full command**

Run: `uv run --package tinfer --extra inference --with matplotlib python -m unittest test_speed.test_benchmark -v`

Run: `./test_speed/run_benchmark.sh`

Expected: both 320-audio profiles complete; both manifests disable speed correction; all four histograms render with shared 0.25-wide bins.

- [ ] **Step 4: Audit, commit, and verify**

```bash
git add test_speed/benchmark_reporting.py test_speed/run_benchmark.py test_speed/test_benchmark.py
git commit -m "feat: benchmark uncorrected predictor speed"
```

Verify 16+ tests pass, 640 WAVs are non-empty, both manifests contain `speed_correction: false`, and all histogram PNG files are non-empty.
