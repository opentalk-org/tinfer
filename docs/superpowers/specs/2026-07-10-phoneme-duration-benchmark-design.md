# Phoneme Duration Benchmark Design

## Goal

Provide one reproducible command that selects 20 reference recordings, embeds them as StyleTTS2 voices, synthesizes unchunked Polish speech, and reports duration-predictor behavior globally and for four selected voices.

## Command and Inputs

The benchmark runs with:

```bash
uv run python test_speed/run_benchmark.py
```

The runner uses `test_speed/archive.zip` and `/workspace/converted_models/magda/model.pth`. A fixed random seed selects 20 WAV files from the archive and four of those voices for individual reports. Sixteen fixed, natural Polish inputs span 2 through 300 characters. Their measured character lengths, rather than nominal labels, are written to every result row.

## Pipeline

The command performs four visible `tqdm` stages:

1. Extract valid WAV references while excluding archive metadata.
2. Encode the selected 20 recordings into StyleTTS2 style vectors.
3. Synthesize every selected voice and text pair, producing 320 WAV files.
4. Aggregate raw measurements and render reports.

The script loads the Magda model once with its style encoder. Synthesis calls `StyleTTS2.generate()` directly instead of the streaming engine or text chunker. Phoneme alignment is requested because the model constructs it from the same rounded `pred_dur` values used to generate the audio. The script asserts that each request stays within one model token window and that no merged-window metadata is present.

## Measurements

For every generated request, the raw result records:

- voice identifier;
- text identifier, text, and character length;
- phoneme count;
- summed predicted duration in seconds;
- predictor-derived phonemes per second;
- generated WAV path.

For every phoneme occurrence, a second raw table records its symbol and predictor-derived duration. The predictor frame duration is the model preprocessing hop length divided by sample rate. The beginning-of-sequence token is excluded because it is not a phoneme.

Global phoneme statistics pool occurrences from all 20 voices. The per-voice statistics use each of four seeded-random voices. Each table contains one row per phoneme symbol with occurrence count and mean, minimum, maximum, p10, and p90 duration. The global scatter includes every generated request; each per-voice scatter includes all 16 requests for that voice. Scatter axes are input character length and predictor-derived phonemes per second.

## Outputs

All generated data lives under `test_speed/results/`:

- `references/`: extracted selected WAV files;
- `embeddings/`: serialized style vectors;
- `audio/`: 320 generated WAV files;
- `metrics/`: raw request and phoneme-occurrence CSV/JSON files;
- `summary/`: global and four per-voice statistic tables, scatter PNG files, and a Markdown index linking the artifacts.

The runner writes a manifest containing the seed, model path, selected voices, highlighted voices, and text corpus so a run can be audited.

## Failure Behavior and Reproducibility

Missing inputs, fewer than 20 valid WAVs, model-window splitting, missing phoneme alignments, and non-positive predicted durations fail clearly. Existing files inside the benchmark-owned results tree are overwritten deterministically. Python, NumPy, and PyTorch random generators use the same fixed seed. CUDA deterministic settings are enabled where supported.

## Verification

Unit tests cover deterministic selection, duration aggregation, percentile tables, and unchunked-result validation without loading the model. A reduced smoke mode exercises extraction, embedding, one synthesis, and report writing. The full run verifies that it creates 20 embeddings, 320 non-empty WAV files, one global report, four per-voice reports, and complete raw records.
